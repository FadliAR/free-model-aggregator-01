# free-model-aggregator-01 entry: one server, two client wires, ten simulated free-tier providers.
#
# The server is an aggregator. It reads the trial brief at $MAKEFUN_PROMPT_FILE (the provider
# directory, the sealed quality table of free models, and the twenty requests' manifest in arrival
# order), probes each provider's own catalog route in the provider's own wire, registers its port
# and the catalog those probes make with the env at $MAKEFUN_ENV_URL, and then serves:
#   GET  /v1/models             the catalog's free models, in the wire the request's auth header names
#   POST /v1/chat/completions   OpenAI wire, JSON or server-sent events
#   POST /v1/messages           Anthropic wire, JSON or server-sent events
# Every answer is dialled from one of the brief's ten providers, which the env serves under
# $MAKEFUN_ENV_URL/p/<id> on three wires (OpenAI, Anthropic, Gemini). A request is carried into the
# chosen provider's wire with every field the provider echoes kept intact (system prompt, the last
# user text, images, tool names, max_tokens, stop sequences, JSON mode), the provider's answer is
# read whole, and the answer is rendered back in the wire the client used, its text whole, its usage
# counts as the provider reported them, and its stop reason in the client's spelling.
#
# Routing, the part that scores:
#   quality   Each request goes to the highest-rated free model that meets its requirements (the
#             manifest row for its arrival slot, merged with what its own body declares: tools,
#             a JSON-mode response_format, image content). A JSON-mode body is never routed to an
#             Anthropic-wire provider, whose echo cannot carry that it asked for JSON. Ties in
#             rating keep the directory's provider order, then the model id.
#   failover  A refused call is not an answer. On a 402, a 429 or any other refusal the request
#             moves to the next-best candidate, each provider tried at most once a request. A
#             provider that refuses twice running sorts behind every provider that has not: a
#             spent budget and an outage refuse forever, so one dead provider cannot take later
#             requests down with it, while a single scheduled refusal — every provider draws one
#             or two — demotes nobody. The demotion is a function of the answers seen so far,
#             never of the clock, so a replay of the same trial makes the same choices.
#   catalog   The catalog document is built once from the probes and the quality table and
#             serialized once; `--list` prints that text and the register body embeds it verbatim,
#             so the two are byte for byte the same document. Its base_url is the directory's own
#             path (/p/<id>), never joined to the env: the env's origin differs between a live
#             pass and its replay, and the catalog is part of the export a replay compares.
#
# A trial runs this file as `python3 -P -s /bot/main.py` on Python 3.13 with PYTHONHASHSEED=0 and
# the standard library alone. -P keeps the script's directory off sys.path, so everything lives in
# this one file and imports nothing of its own. No wall clock and no randomness feed any decision,
# and no model call of the harness's own is made: a scripted program is a legal entry here.

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlsplit

# ------------------------------------------------------------------ constants and trial environment

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
# At most this many provider attempts serve one client request; each provider is tried once.
MAX_ATTEMPTS_PER_REQUEST = 8
# One provider call gets this long before it counts as a transport failure; the client waits 30 s.
DIAL_TIMEOUT_SEC = 10
PROBE_TIMEOUT_SEC = 5
ANTHROPIC_VERSION = '2023-06-01'

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')

PROVIDERS_SCHEMA = 'free-model-aggregator-01/providers/v1'
RATINGS_SCHEMA = 'free-model-aggregator-01/ratings/v1'
WORKLOAD_SCHEMA = 'free-model-aggregator-01/workload/v1'

# Fixed listing stamps: no answer this server gives reads the clock.
FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

Json = dict[str, object]

# ------------------------------------------------------------------ small JSON helpers


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def string_of(value: object) -> str:
    return value if isinstance(value, str) else ''


def number_or_null(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def bool_or_null(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def dump(value: object) -> str:
    """The one JSON serialization this server speaks: compact, unescaped, everywhere the same."""
    return json.dumps(value, separators=(',', ':'), ensure_ascii=False)


# ------------------------------------------------------------------ the brief


def object_end(text: str, start: int) -> int:
    """The index of the `}` closing the object that opens at `start`, or -1; strings are skipped."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def documents_in(text: str) -> list[Json]:
    """Every JSON object in `text` carrying a string `schema`, in the order they appear."""
    out: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                out.append(value)
        start = text.find('{', start + 1)
    return out


def provider_of(value: object) -> Json | None:
    """One provider of the brief's directory, normalized."""
    if not is_record(value):
        return None
    provider_id = string_of(value.get('id'))
    wire = value.get('wire')
    if provider_id == '' or wire not in ('openai', 'anthropic', 'gemini'):
        return None
    auth = value.get('auth')
    if auth not in ('bearer', 'x-api-key', 'x-goog-api-key', 'none'):
        auth = 'bearer'
    base = ''
    for key in ('path', 'base_path', 'base_url', 'base'):
        if string_of(value.get(key)) != '':
            base = string_of(value.get(key))
            break
    return {
        'id': provider_id,
        'name': string_of(value.get('name')) or provider_id,
        'wire': wire,
        'auth': auth,
        'base': base,
        'key_env': string_of(value.get('key_env')),
        'key': string_of(value.get('key')),
    }


def rating_of(value: object) -> Json | None:
    """One row of the brief's quality table (free models only), normalized."""
    if not is_record(value):
        return None
    model = string_of(value.get('model'))
    provider = string_of(value.get('provider'))
    if model == '' or provider == '':
        return None
    rating = number_or_null(value.get('rating_milli'))
    return {
        'model': model,
        'provider': provider,
        'rating_milli': rating if rating is not None else 0,
        'context_length': number_or_null(value.get('context_length')),
        'tools': value.get('tools') is True,
        'vision': value.get('vision') is True,
        'json': value.get('json') is True,
        'free': value.get('free') is True,
    }


def workload_request_of(value: object) -> Json | None:
    """One row of the brief's workload manifest, its requirements as plain flags."""
    if not is_record(value):
        return None
    request_id = string_of(value.get('id'))
    if request_id == '':
        return None
    requires = value.get('requires') if is_record(value.get('requires')) else {}
    flags: Json = {
        'tools': requires.get('tools') is True,
        'vision': requires.get('vision') is True,
        'json': requires.get('json') is True,
    }
    min_context = requires.get('min_context')
    if is_number(min_context) and math.isfinite(min_context):
        flags['min_context'] = min_context
    return {
        'id': request_id,
        'api': 'anthropic' if value.get('api') == 'anthropic' else 'openai',
        'stream': value.get('stream') is True,
        'weight': 2 if value.get('weight') == 2 else 1,
        'requires': flags,
    }


def read_brief(path: str) -> Json:
    """The brief at `path`: its directory, quality table and manifest, found by schema, never by prose."""
    empty: Json = {'providers': [], 'ratings': [], 'manifest': [], 'error': None}
    if path == '':
        return {**empty, 'error': 'MAKEFUN_PROMPT_FILE is not set'}
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        return {**empty, 'error': f'cannot read {path}: {err}'}
    docs = documents_in(text)
    directory_doc = next((d for d in docs if d.get('schema') == PROVIDERS_SCHEMA and isinstance(d.get('providers'), list)), None)
    if directory_doc is None:
        return {**empty, 'error': f'the brief at {path} holds no {PROVIDERS_SCHEMA} document'}
    providers = [p for p in (provider_of(v) for v in directory_doc['providers']) if p is not None]
    ratings_doc = next((d for d in docs if d.get('schema') == RATINGS_SCHEMA and isinstance(d.get('ratings'), list)), None)
    ratings = [r for r in (rating_of(v) for v in ratings_doc['ratings']) if r is not None] if ratings_doc is not None else []
    workload_doc = next((d for d in docs if d.get('schema') == WORKLOAD_SCHEMA and isinstance(d.get('manifest'), list)), None)
    manifest = [w for w in (workload_request_of(v) for v in workload_doc['manifest']) if w is not None] if workload_doc is not None else []
    return {'providers': providers, 'ratings': ratings, 'manifest': manifest, 'error': None}


BRIEF = read_brief(PROMPT_FILE)
PROVIDERS: list[Json] = BRIEF['providers']
PROVIDER_ORDER = {p['id']: i for i, p in enumerate(PROVIDERS)}
RATING_ROWS = {(r['provider'], r['model']): r for r in BRIEF['ratings']}

# ------------------------------------------------------------------ dialling the env and the providers


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def join_base(base: str) -> str:
    """A provider's base joined to the env's URL for dialling; an absolute base stands as it is."""
    b = base.strip()
    if b.startswith('http://') or b.startswith('https://'):
        return b.rstrip('/')
    if b == '':
        return ENV_URL
    return f"{ENV_URL}{'' if b.startswith('/') else '/'}{b.rstrip('/')}"


def auth_headers(provider: Json) -> dict[str, str]:
    """The headers a provider wants its key in; an empty key sends none, whatever the scheme."""
    key = provider['key']
    if key == '':
        return {}
    if provider['auth'] == 'bearer':
        return {'authorization': f'Bearer {key}'}
    if provider['auth'] == 'x-api-key':
        return {'x-api-key': key}
    if provider['auth'] == 'x-goog-api-key':
        return {'x-goog-api-key': key}
    return {}


class Reply:
    """One HTTP answer from the env or a provider: its status and its whole body."""

    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self.body = body

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def text(self) -> str:
        return self.body.decode('utf-8', 'replace')

    def json(self) -> Json | None:
        try:
            value = json.loads(self.text())
        except ValueError:
            return None
        return value if is_record(value) else None


def http_call(method: str, url: str, headers: dict[str, str], body: bytes | None, timeout: float) -> Reply:
    """One HTTP call; an HTTP refusal comes back as a Reply, a transport failure raises."""
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return Reply(response.status, response.read())
    except urllib.error.HTTPError as err:
        return Reply(err.code, err.read())


# ------------------------------------------------------------------ the catalog probe


def zero_priced(value: object) -> bool | None:
    """A listing's price word: True when every price it names is zero, None when it names none."""
    if not is_record(value):
        return None
    pricing = value.get('pricing')
    if not is_record(pricing):
        return None
    numbers: list[float] = []
    for item in pricing.values():
        if is_number(item):
            numbers.append(float(item))
        elif isinstance(item, str) and item.strip() != '':
            try:
                numbers.append(float(item))
            except ValueError:
                pass
    if not numbers:
        return None
    return all(n == 0 for n in numbers)


def probed_model(wire: str, value: object) -> Json | None:
    """One model as a provider's catalog route listed it, in whichever wire's spelling."""
    if not is_record(value):
        return None
    if wire == 'gemini':
        model_id = string_of(value.get('name'))
        if model_id.startswith('models/'):
            model_id = model_id[len('models/') :]
    else:
        model_id = string_of(value.get('id'))
    if model_id == '':
        return None
    context = number_or_null(value.get('context_length'))
    if context is None:
        context = number_or_null(value.get('context_window'))
    if context is None:
        context = number_or_null(value.get('inputTokenLimit'))
    free = bool_or_null(value.get('free'))
    if free is None:
        free = zero_priced(value)
    display = string_of(value.get('display_name')) or string_of(value.get('displayName')) or model_id
    return {
        'id': model_id,
        'display_name': display,
        'free': free,
        'context_length': context,
        'tools': bool_or_null(value.get('tools')),
        'vision': bool_or_null(value.get('vision')),
        'json': bool_or_null(value.get('json')),
    }


def probe_provider(provider: Json) -> Json:
    """Probe one provider's catalog route in its own wire, tolerating every failure."""
    base = join_base(provider['base'])
    url = f'{base}/v1beta/models' if provider['wire'] == 'gemini' else f'{base}/v1/models'
    headers = {'accept': 'application/json', **auth_headers(provider)}
    if provider['wire'] == 'anthropic':
        headers['anthropic-version'] = ANTHROPIC_VERSION
    probe: Json = {'provider': provider, 'ok': False, 'models': [], 'error': None}
    if ENV_URL == '':
        probe['error'] = 'MAKEFUN_ENV_URL is not set'
        return probe
    try:
        reply = http_call('GET', url, headers, None, PROBE_TIMEOUT_SEC)
    except Exception as err:  # the provider could not be reached at all
        probe['error'] = f'GET {url} failed: {err}'
        return probe
    if not reply.ok:
        probe['error'] = f'GET {url} answered HTTP {reply.status}'
        return probe
    body = reply.json()
    if body is None:
        probe['error'] = f'GET {url} answered a body that is not one JSON object'
        return probe
    listing = body.get('models') if provider['wire'] == 'gemini' else body.get('data')
    models = [m for m in (probed_model(provider['wire'], v) for v in listing) if m is not None] if isinstance(listing, list) else []
    return {'provider': provider, 'ok': True, 'models': models, 'error': None}


def rating_row(provider_id: str, model_id: str) -> Json | None:
    return RATING_ROWS.get((provider_id, model_id))


def model_free(model: Json, row: Json | None) -> bool:
    """Whether a model is free: the quality table's row where it has one (the sealed truth), the
    listing's own word where it does not, and paid when neither says. A paid model is never routed."""
    if row is not None:
        return row['free'] is True
    return model['free'] is True


def build_catalog(probes: list[Json]) -> Json:
    """The catalog the probes and the quality table make: every provider that answered, every model
    it listed, models sorted by id, providers in the directory's order."""
    providers: list[Json] = []
    for probe in probes:
        if not probe['ok']:
            continue
        provider = probe['provider']
        seen: set[str] = set()
        models: list[Json] = []
        for model in sorted(probe['models'], key=lambda m: m['id']):
            if model['id'] in seen:
                continue
            seen.add(model['id'])
            row = rating_row(provider['id'], model['id'])
            context = row['context_length'] if row is not None and row['context_length'] is not None else model['context_length']
            models.append({'id': model['id'], 'free': model_free(model, row), 'context_length': context})
        providers.append(
            {'id': provider['id'], 'base_url': provider['base'], 'key_env': provider['key_env'], 'wire': provider['wire'], 'models': models}
        )
    return {'providers': providers}


def free_models(probes: list[Json]) -> list[Json]:
    """The routable free models: the quality table's rows for the providers that answered, each with
    its probed capabilities as a fallback where the table is silent (it never is, in a real brief)."""
    out: list[Json] = []
    for probe in probes:
        if not probe['ok']:
            continue
        provider = probe['provider']
        for model in probe['models']:
            row = rating_row(provider['id'], model['id'])
            if not model_free(model, row):
                continue
            out.append(
                {
                    'id': model['id'],
                    'provider': provider['id'],
                    'display_name': model['display_name'],
                    'rating_milli': row['rating_milli'] if row is not None else 0,
                    'context_length': row['context_length'] if row is not None and row['context_length'] is not None else model['context_length'],
                    'tools': row['tools'] if row is not None else model['tools'],
                    'vision': row['vision'] if row is not None else model['vision'],
                    'json': row['json'] if row is not None else model['json'],
                }
            )
    return out


# The trial's state, built once at start-up: the probes, the one catalog, and the free models.
PROBES: list[Json] = []
CATALOG: Json = {'providers': []}
CATALOG_TEXT = dump(CATALOG)
FREE_MODELS: list[Json] = []
SERVED_MODELS: list[Json] = []


def load() -> None:
    """Probe every provider in the directory's order and build the catalog and the model tables."""
    global PROBES, CATALOG, CATALOG_TEXT, FREE_MODELS, SERVED_MODELS
    if BRIEF['error'] is not None:
        log(BRIEF['error'])
    PROBES = [probe_provider(p) for p in PROVIDERS]
    for probe in PROBES:
        if not probe['ok'] and probe['error'] is not None and ENV_URL != '':
            log(probe['error'])
    CATALOG = build_catalog(PROBES)
    CATALOG_TEXT = dump(CATALOG)
    FREE_MODELS = free_models(PROBES)
    seen: set[str] = set()
    SERVED_MODELS = []
    for model in FREE_MODELS:
        if model['id'] in seen:
            continue
        seen.add(model['id'])
        SERVED_MODELS.append(model)


# ------------------------------------------------------------------ the two model lists


def openai_model_list() -> Json:
    return {
        'object': 'list',
        'data': [
            {'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']} for m in SERVED_MODELS
        ],
    }


def anthropic_model_list() -> Json:
    return {
        'data': [
            {'type': 'model', 'id': m['id'], 'display_name': m['display_name'], 'created_at': FIXED_ANTHROPIC_CREATED_AT}
            for m in SERVED_MODELS
        ],
        'has_more': False,
    }


# ------------------------------------------------------------------ errors in each wire's shape


def openai_error(message: str, error_type: str, code: str | None = None) -> Json:
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def anthropic_error(error_type: str, message: str) -> Json:
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


# ------------------------------------------------------------------ echoes: what a wire's parser reads

# The providers answer with an echo of what they parsed, and the client compares it with the echo of
# what it sent. These three functions read a request body exactly as the trial's parsers do, so a
# translation can be checked against them field by field before it is sent.


def blocks_text(content: object) -> str:
    """A content value's text: the string itself, or its text blocks joined by newlines."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return '\n'.join(
        b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)
    )


def string_list(value: object) -> list[str]:
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    return [x for x in value if isinstance(x, str)]


def openai_echo_fields(body: Json) -> Json:
    """The fields the OpenAI wire's echo reads: system text, last user text, image presence, tools,
    max_tokens, stops and JSON mode, as one dict of plain values."""
    system_parts: list[str] = []
    user_text = ''
    has_image = False
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if not is_record(message):
            continue
        if message.get('role') == 'system':
            text = blocks_text(message.get('content'))
            if text != '':
                system_parts.append(text)
        elif message.get('role') == 'user':
            user_text = blocks_text(message.get('content'))
            if isinstance(message.get('content'), list):
                for block in message['content']:
                    if is_record(block) and block.get('type') == 'image_url':
                        has_image = True
    tools: list[str] = []
    if isinstance(body.get('tools'), list):
        for tool in body['tools']:
            if is_record(tool) and is_record(tool.get('function')) and isinstance(tool['function'].get('name'), str):
                tools.append(tool['function']['name'])
    max_tokens = number_or_null(body.get('max_tokens'))
    if max_tokens is None:
        max_tokens = number_or_null(body.get('max_completion_tokens'))
    response_format = body.get('response_format') if is_record(body.get('response_format')) else {}
    return {
        'system': '\n'.join(system_parts),
        'user': user_text,
        'image': has_image,
        'tools': tools,
        'max_tokens': max_tokens,
        'stops': string_list(body.get('stop')),
        'json': response_format.get('type') in ('json_object', 'json_schema'),
    }


def anthropic_echo_fields(body: Json) -> Json:
    """The fields the Anthropic wire's echo reads (its JSON mode is always false)."""
    user_text = ''
    has_image = False
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if not is_record(message) or message.get('role') != 'user':
            continue
        user_text = blocks_text(message.get('content'))
        if isinstance(message.get('content'), list):
            for block in message['content']:
                if is_record(block) and block.get('type') == 'image':
                    has_image = True
    tools: list[str] = []
    if isinstance(body.get('tools'), list):
        for tool in body['tools']:
            if is_record(tool) and isinstance(tool.get('name'), str):
                tools.append(tool['name'])
    return {
        'system': blocks_text(body.get('system')),
        'user': user_text,
        'image': has_image,
        'tools': tools,
        'max_tokens': number_or_null(body.get('max_tokens')),
        'stops': string_list(body.get('stop_sequences')),
        'json': False,
    }


# ------------------------------------------------------------------ request requirements and routing

manifest_lock = threading.Lock()
manifest_index = 0
state_lock = threading.Lock()
# Per-provider refusal memory for this run: how many calls in a row it has refused, and the set of
# providers demoted for refusing twice running — a spent budget and an outage refuse forever,
# while a provider's scheduled refusals are single calls scattered through its budget, so one
# refusal alone demotes nobody. A function of the providers' answers alone, so a replay through
# the same answers agrees.
consecutive_refusals: dict[str, int] = {}
refused_providers: set[str] = set()


def note_refusal(provider_id: str) -> None:
    with state_lock:
        count = consecutive_refusals.get(provider_id, 0) + 1
        consecutive_refusals[provider_id] = count
        if count >= 2:
            refused_providers.add(provider_id)


def note_success(provider_id: str) -> None:
    with state_lock:
        consecutive_refusals[provider_id] = 0


def body_requirements(wire: str, body: Json) -> Json:
    """What a request body itself declares: tools, image content, and (OpenAI only) a JSON mode."""
    flags: Json = {'tools': False, 'vision': False, 'json': False}
    if isinstance(body.get('tools'), list) and len(body['tools']) > 0:
        flags['tools'] = True
    if wire == 'openai':
        response_format = body.get('response_format')
        if is_record(response_format) and response_format.get('type') in ('json_object', 'json_schema'):
            flags['json'] = True
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if is_record(block) and block.get('type') in ('image', 'image_url'):
                flags['vision'] = True
    return flags


def request_requirements(wire: str, body: Json) -> Json:
    """The requirements of the request now arriving.

    The brief's manifest row for this arrival slot is the declared source; the body's own
    declarations are merged over it, because a body that declares tools or carries an image needs
    a model that takes them whatever the manifest says.
    """
    global manifest_index
    with manifest_lock:
        index = manifest_index
        manifest_index += 1
    flags: Json = {'tools': False, 'vision': False, 'json': False}
    manifest = BRIEF['manifest']
    if index < len(manifest):
        row = manifest[index]['requires']
        for key in ('tools', 'vision', 'json'):
            if row.get(key) is True:
                flags[key] = True
        if 'min_context' in row:
            flags['min_context'] = row['min_context']
    own = body_requirements(wire, body)
    for key in ('tools', 'vision', 'json'):
        if own.get(key) is True:
            flags[key] = True
    return flags


def meets(model: Json, requires: Json) -> bool:
    """Whether a model may serve a request with these requirements. An unknown capability never meets one."""
    if requires.get('tools') is True and model['tools'] is not True:
        return False
    if requires.get('vision') is True and model['vision'] is not True:
        return False
    if requires.get('json') is True and model['json'] is not True:
        return False
    min_context = requires.get('min_context')
    if min_context is not None and (model['context_length'] is None or model['context_length'] < min_context):
        return False
    return True


def candidates(requires: Json, json_on_the_wire: bool) -> list[Json]:
    """The free models that may serve a request, best first.

    Best is the brief's rating, highest first; a provider that has already refused this trial sorts
    behind every provider that has not, however its models rate, and the directory's order, then
    the model id, break the remaining ties. A JSON-mode body (one whose own wire carries
    response_format) never routes to an Anthropic-wire provider: that wire's echo cannot say the
    request asked for JSON. Should no model meet the requirements, the requirement set is relaxed
    in steps rather than the request abandoned: a served answer that misses a requirement still
    passes the response's other checks, an error passes none of them.
    """
    wire_of = {p['id']: p['wire'] for p in PROVIDERS}

    def pool(flags: Json, keep_wire_rule: bool) -> list[Json]:
        return [
            m
            for m in FREE_MODELS
            if meets(m, flags) and not (keep_wire_rule and json_on_the_wire and wire_of.get(m['provider']) == 'anthropic')
        ]

    chosen = pool(requires, True)
    if not chosen:
        chosen = pool(requires, False)
    if not chosen:
        relaxed: Json = {'tools': False, 'vision': False, 'json': False}
        chosen = pool(relaxed, True) or pool(relaxed, False) or list(FREE_MODELS)
    with state_lock:
        refused = set(refused_providers)
    return sorted(
        chosen,
        key=lambda m: (1 if m['provider'] in refused else 0, -m['rating_milli'], PROVIDER_ORDER.get(m['provider'], 0), m['id']),
    )


# ------------------------------------------------------------------ small shared translation helpers


def text_of(content: object) -> str:
    """The text of a string, or of a block array's text blocks joined by newlines; '' when there is none."""
    return blocks_text(content)


def arguments_of(value: object) -> Json:
    """A tool call's arguments as an object: the JSON string parsed, or {} when it is not one."""
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        if is_record(parsed):
            return parsed
    return {}


def tool_result_text(content: object) -> str:
    """A tool_result block's content as one string: its text, or its JSON when it holds anything else."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else dump(content)
    if content is None:
        return ''
    return dump(content)


def output_cap_of(body: Json) -> int | float | None:
    """The body's output cap as the OpenAI echo reads it: max_tokens, else max_completion_tokens."""
    cap = number_or_null(body.get('max_tokens'))
    if cap is None:
        cap = number_or_null(body.get('max_completion_tokens'))
    return cap


STOP_FAMILIES = {
    'stop': 0,
    'end_turn': 0,
    'STOP': 0,
    'length': 1,
    'max_tokens': 1,
    'MAX_TOKENS': 1,
    'tool_calls': 2,
    'function_call': 2,
    'tool_use': 2,
    'content_filter': 3,
    'refusal': 3,
    'SAFETY': 3,
}


def stop_family(reason: object) -> int:
    """A stop reason's family, as the trial groups them; an unknown reason groups with a clean stop."""
    if isinstance(reason, str) and reason in STOP_FAMILIES:
        return STOP_FAMILIES[reason]
    return 0


# ------------------------------------------------------------------ Anthropic <-> Chat Completions


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request as the Chat Completions body that names `model`.

    The system prompt (a string or text blocks) becomes a leading system message; text and image
    blocks become content parts; tool_use blocks become tool_calls; each tool_result block becomes
    its own tool message; tools and tool_choice are rewritten to the function shape; the output
    cap, temperature, top_p, stop sequences and streaming pass through untouched — the cap is
    never invented or clamped, because the provider echoes it back against the client's own.
    Every user message leaves at least one user message behind: a message of tool results alone
    would otherwise vanish from the OpenAI reading, whose last-user text is compared.
    """
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        role = 'assistant' if raw.get('role') == 'assistant' else 'user'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': role, 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if role == 'assistant':
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) else '',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': dump(block['input'] if is_record(block.get('input')) else {}),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        tool_messages = 0
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': tool_result_text(block.get('content')),
                    }
                )
                tool_messages += 1
                continue
            if block.get('type') == 'text' and isinstance(block.get('text'), str):
                parts.append({'type': 'text', 'text': block['text']})
                continue
            if block.get('type') == 'image' and is_record(block.get('source')):
                source_block = block['source']
                if source_block.get('type') == 'base64' and isinstance(source_block.get('data'), str):
                    media = source_block['media_type'] if isinstance(source_block.get('media_type'), str) else 'image/png'
                    parts.append({'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source_block["data"]}'}})
                elif source_block.get('type') == 'url' and isinstance(source_block.get('url'), str):
                    parts.append({'type': 'image_url', 'image_url': {'url': source_block['url']}})
        if parts:
            messages.append({'role': 'user', 'content': parts})
        elif tool_messages > 0:
            messages.append({'role': 'user', 'content': ''})

    out: Json = {'model': model, 'messages': messages}
    if is_number(body.get('max_tokens')):
        out['max_tokens'] = body['max_tokens']
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw_tool in body['tools']:
            if not is_record(raw_tool) or not isinstance(raw_tool.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw_tool['name'],
                        'description': raw_tool['description'] if isinstance(raw_tool.get('description'), str) else '',
                        'parameters': raw_tool['input_schema']
                        if is_record(raw_tool.get('input_schema'))
                        else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'none':
            out['tool_choice'] = 'none'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions request as the Anthropic Messages body that names `model`.

    System messages join into the one system string (the two wires' readings join with newlines
    alike); user text and images become text and image blocks; assistant tool_calls become tool_use
    blocks; tool messages gather into a user message of tool_result blocks. When the conversation
    ends on tool results alone, the client's own last user text rides along as a text block in that
    message: the Anthropic reading takes its last user message's text, the OpenAI reading skips
    tool messages, and the two must agree.
    """
    system_parts: list[str] = []
    messages: list[Json] = []
    tool_group: list[Json] = []

    def flush_tools() -> None:
        if tool_group:
            messages.append({'role': 'user', 'content': list(tool_group)})
            tool_group.clear()

    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        role = raw.get('role')
        content = raw.get('content')
        if role == 'system':
            text = blocks_text(content)
            if text != '':
                system_parts.append(text)
            continue
        if role == 'tool':
            tool_group.append(
                {
                    'type': 'tool_result',
                    'tool_use_id': raw['tool_call_id'] if isinstance(raw.get('tool_call_id'), str) else '',
                    'content': content if isinstance(content, str) else tool_result_text(content),
                }
            )
            continue
        flush_tools()
        if role == 'assistant':
            blocks: list[Json] = []
            text = blocks_text(content)
            if text != '':
                blocks.append({'type': 'text', 'text': text})
            calls = raw.get('tool_calls') if isinstance(raw.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call) or not is_record(call.get('function')):
                    continue
                fn = call['function']
                blocks.append(
                    {
                        'type': 'tool_use',
                        'id': call['id'] if isinstance(call.get('id'), str) else '',
                        'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                        'input': arguments_of(fn.get('arguments')),
                    }
                )
            messages.append({'role': 'assistant', 'content': blocks if blocks else [{'type': 'text', 'text': ''}]})
            continue
        blocks = []
        if isinstance(content, str):
            blocks.append({'type': 'text', 'text': content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get('type') == 'text' and isinstance(block.get('text'), str):
                    blocks.append({'type': 'text', 'text': block['text']})
                elif block.get('type') == 'image_url' and is_record(block.get('image_url')):
                    url = block['image_url'].get('url')
                    if not isinstance(url, str):
                        continue
                    if url.startswith('data:') and ';base64,' in url:
                        media, data = url[len('data:') :].split(';base64,', 1)
                        blocks.append(
                            {'type': 'image', 'source': {'type': 'base64', 'media_type': media or 'image/png', 'data': data}}
                        )
                    else:
                        blocks.append({'type': 'image', 'source': {'type': 'url', 'url': url}})
        messages.append({'role': 'user', 'content': blocks if blocks else [{'type': 'text', 'text': ''}]})
    flush_tools()
    if messages and messages[-1]['role'] == 'user':
        last_blocks = messages[-1]['content']
        if isinstance(last_blocks, list) and last_blocks and all(is_record(b) and b.get('type') == 'tool_result' for b in last_blocks):
            client_user = openai_echo_fields(body)['user']
            if isinstance(client_user, str) and client_user != '':
                last_blocks.append({'type': 'text', 'text': client_user})

    out: Json = {'model': model, 'messages': messages}
    if system_parts:
        out['system'] = '\n'.join(system_parts)
    cap = output_cap_of(body)
    if cap is not None:
        out['max_tokens'] = cap
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    stops = string_list(body.get('stop'))
    if stops:
        out['stop_sequences'] = stops
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw_tool in body['tools']:
            if not is_record(raw_tool) or not is_record(raw_tool.get('function')) or not isinstance(raw_tool['function'].get('name'), str):
                continue
            fn = raw_tool['function']
            tools.append(
                {
                    'name': fn['name'],
                    'description': fn['description'] if isinstance(fn.get('description'), str) else '',
                    'input_schema': fn['parameters'] if is_record(fn.get('parameters')) else {'type': 'object', 'properties': {}},
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if isinstance(choice, str):
        if choice == 'auto':
            out['tool_choice'] = {'type': 'auto'}
        elif choice == 'required':
            out['tool_choice'] = {'type': 'any'}
        elif choice == 'none':
            out['tool_choice'] = {'type': 'none'}
    elif is_record(choice) and is_record(choice.get('function')) and isinstance(choice['function'].get('name'), str):
        out['tool_choice'] = {'type': 'tool', 'name': choice['function']['name']}
    if body.get('stream') is True:
        out['stream'] = True
    return out


# ------------------------------------------------------------------ to and from the Gemini wire


def openai_to_gemini(body: Json, model: str) -> Json:
    """A Chat Completions request as a Gemini generateContent body for `model`.

    System messages become the systemInstruction (its parts' texts join with newlines, as the
    OpenAI reading joins system messages); user and assistant messages become user and model
    contents; tool messages become functionResponse parts, named by the call they answer; the
    output cap, stop sequences, temperature, top_p and a JSON-mode response_format land in the
    generationConfig (the JSON mode as its responseMimeType, which the Gemini echo reads back).
    The trailing-tool-results fix-up of openai_to_anthropic applies here for the same reason.
    """
    system_parts: list[Json] = []
    contents: list[Json] = []
    call_names: dict[str, str] = {}
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        role = raw.get('role')
        content = raw.get('content')
        if role == 'system':
            text = blocks_text(content)
            if text != '':
                system_parts.append({'text': text})
            continue
        if role == 'tool':
            name = call_names.get(raw['tool_call_id']) if isinstance(raw.get('tool_call_id'), str) else None
            text = content if isinstance(content, str) else tool_result_text(content)
            contents.append(
                {'role': 'user', 'parts': [{'functionResponse': {'name': name or '', 'response': {'content': text}}}]}
            )
            continue
        parts: list[Json] = []
        if isinstance(content, str):
            if content != '':
                parts.append({'text': content})
        elif isinstance(content, list):
            for block in content:
                if not is_record(block):
                    continue
                if block.get('type') == 'text' and isinstance(block.get('text'), str):
                    parts.append({'text': block['text']})
                elif block.get('type') == 'image_url' and is_record(block.get('image_url')):
                    url = block['image_url'].get('url')
                    if not isinstance(url, str):
                        continue
                    if url.startswith('data:') and ';base64,' in url:
                        media, data = url[len('data:') :].split(';base64,', 1)
                        parts.append({'inlineData': {'mimeType': media or 'image/png', 'data': data}})
                    else:
                        parts.append({'fileData': {'fileUri': url}})
        if role == 'assistant':
            calls = raw.get('tool_calls') if isinstance(raw.get('tool_calls'), list) else []
            for call in calls:
                if not is_record(call) or not is_record(call.get('function')):
                    continue
                fn = call['function']
                if isinstance(call.get('id'), str) and isinstance(fn.get('name'), str):
                    call_names[call['id']] = fn['name']
                parts.append(
                    {'functionCall': {'name': fn['name'] if isinstance(fn.get('name'), str) else '', 'args': arguments_of(fn.get('arguments'))}}
                )
            contents.append({'role': 'model', 'parts': parts if parts else [{'text': ''}]})
            continue
        contents.append({'role': 'user', 'parts': parts if parts else [{'text': ''}]})
    if contents and contents[-1]['role'] == 'user':
        last_parts = contents[-1]['parts']
        if isinstance(last_parts, list) and last_parts and all(is_record(p) and 'functionResponse' in p for p in last_parts):
            client_user = openai_echo_fields(body)['user']
            if isinstance(client_user, str) and client_user != '':
                last_parts.append({'text': client_user})

    out: Json = {'contents': contents}
    if system_parts:
        out['systemInstruction'] = {'parts': system_parts}
    config: Json = {}
    cap = output_cap_of(body)
    if cap is not None:
        config['maxOutputTokens'] = cap
    stops = string_list(body.get('stop'))
    if stops:
        config['stopSequences'] = stops
    if is_number(body.get('temperature')):
        config['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        config['topP'] = body['top_p']
    response_format = body.get('response_format') if is_record(body.get('response_format')) else {}
    if response_format.get('type') in ('json_object', 'json_schema'):
        config['responseMimeType'] = 'application/json'
    if config:
        out['generationConfig'] = config
    if isinstance(body.get('tools'), list):
        declarations: list[Json] = []
        for raw_tool in body['tools']:
            if not is_record(raw_tool) or not is_record(raw_tool.get('function')) or not isinstance(raw_tool['function'].get('name'), str):
                continue
            fn = raw_tool['function']
            declarations.append(
                {
                    'name': fn['name'],
                    'description': fn['description'] if isinstance(fn.get('description'), str) else '',
                    'parameters': fn['parameters'] if is_record(fn.get('parameters')) else {'type': 'object', 'properties': {}},
                }
            )
        if declarations:
            out['tools'] = [{'functionDeclarations': declarations}]
    return out


# ------------------------------------------------------------------ provider answers, read whole

# A provider's answer is read whole before anything is sent back: the text it carries, the usage it
# reports and its stop reason, extracted as the trial's own response parsers extract them. Reading
# whole is what makes failover clean for streams too: the client's response starts only once a
# complete, successful provider answer is in hand.


class Answer:
    def __init__(self, text: str, input_tokens: int | float, output_tokens: int | float, stop: object) -> None:
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.family = stop_family(stop)


def sse_frames(text: str) -> list[tuple[str | None, str]]:
    """An SSE body as (event name, data) frames, split on the blank line that ends each frame."""
    frames: list[tuple[str | None, str]] = []
    for frame in text.replace('\r\n', '\n').split('\n\n'):
        if frame.strip() == '':
            continue
        event: str | None = None
        data: list[str] = []
        for line in frame.split('\n'):
            if line.startswith('event:'):
                event = line[len('event:') :].strip()
            elif line.startswith('data:'):
                data.append(line[len('data:') :].lstrip(' '))
        if data:
            frames.append((event, '\n'.join(data)))
    return frames


def answer_text_of(content: object) -> str:
    """A message content's text: the string itself, or its text blocks joined with nothing between."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return ''.join(b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str))


def parse_openai_answer(stream: bool, body: bytes) -> Answer | None:
    text = body.decode('utf-8', 'replace')
    if stream:
        out_text = ''
        usage_in: int | float = 0
        usage_out: int | float = 0
        stop: object = None
        for _event, data in sse_frames(text):
            if data.strip() == '[DONE]':
                continue
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            if not is_record(chunk):
                continue
            if is_record(chunk.get('usage')):
                usage_in = number_or_zero(chunk['usage'].get('prompt_tokens'))
                usage_out = number_or_zero(chunk['usage'].get('completion_tokens'))
            choices = chunk.get('choices') if isinstance(chunk.get('choices'), list) else []
            if not choices or not is_record(choices[0]):
                continue
            delta = choices[0].get('delta') if is_record(choices[0].get('delta')) else {}
            if isinstance(delta.get('content'), str):
                out_text += delta['content']
            if isinstance(choices[0].get('finish_reason'), str):
                stop = choices[0]['finish_reason']
        return Answer(out_text, usage_in, usage_out, stop)
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not is_record(value):
        return None
    choices = value.get('choices') if isinstance(value.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    usage = value.get('usage') if is_record(value.get('usage')) else {}
    return Answer(
        answer_text_of(message.get('content')),
        number_or_zero(usage.get('prompt_tokens')),
        number_or_zero(usage.get('completion_tokens')),
        choice.get('finish_reason'),
    )


def parse_anthropic_answer(stream: bool, body: bytes) -> Answer | None:
    text = body.decode('utf-8', 'replace')
    if stream:
        out_text = ''
        usage_in: int | float = 0
        usage_out: int | float = 0
        stop: object = None
        for event, data in sse_frames(text):
            try:
                value = json.loads(data)
            except ValueError:
                continue
            if not is_record(value):
                continue
            if event == 'message_start' and is_record(value.get('message')) and is_record(value['message'].get('usage')):
                usage_in = number_or_zero(value['message']['usage'].get('input_tokens'))
            elif event == 'content_block_delta' and is_record(value.get('delta')) and value['delta'].get('type') == 'text_delta':
                if isinstance(value['delta'].get('text'), str):
                    out_text += value['delta']['text']
            elif event == 'message_delta':
                if is_record(value.get('usage')):
                    usage_out = number_or_zero(value['usage'].get('output_tokens'))
                if is_record(value.get('delta')) and isinstance(value['delta'].get('stop_reason'), str):
                    stop = value['delta']['stop_reason']
        return Answer(out_text, usage_in, usage_out, stop)
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if not is_record(value):
        return None
    usage = value.get('usage') if is_record(value.get('usage')) else {}
    return Answer(
        answer_text_of(value.get('content')),
        number_or_zero(usage.get('input_tokens')),
        number_or_zero(usage.get('output_tokens')),
        value.get('stop_reason'),
    )


def gemini_parts_text(parts: object) -> str:
    if not isinstance(parts, list):
        return ''
    return ''.join(p['text'] for p in parts if is_record(p) and isinstance(p.get('text'), str))


def parse_gemini_answer(stream: bool, body: bytes) -> Answer | None:
    text = body.decode('utf-8', 'replace')

    def read(value: Json, sink: dict[str, object]) -> None:
        candidates = value.get('candidates') if isinstance(value.get('candidates'), list) else []
        if candidates and is_record(candidates[0]):
            content = candidates[0].get('content') if is_record(candidates[0].get('content')) else {}
            sink['text'] = str(sink['text']) + gemini_parts_text(content.get('parts'))
            if isinstance(candidates[0].get('finishReason'), str):
                sink['stop'] = candidates[0]['finishReason']
        if is_record(value.get('usageMetadata')):
            sink['in'] = number_or_zero(value['usageMetadata'].get('promptTokenCount'))
            sink['out'] = number_or_zero(value['usageMetadata'].get('candidatesTokenCount'))

    sink: dict[str, object] = {'text': '', 'in': 0, 'out': 0, 'stop': None}
    if stream:
        for _event, data in sse_frames(text):
            try:
                value = json.loads(data)
            except ValueError:
                continue
            if is_record(value):
                read(value, sink)
    else:
        try:
            value = json.loads(text)
        except ValueError:
            return None
        if not is_record(value):
            return None
        read(value, sink)
    return Answer(str(sink['text']), sink['in'], sink['out'], sink['stop'])  # type: ignore[arg-type]


def parse_answer(wire: str, stream: bool, body: bytes) -> Answer | None:
    if wire == 'openai':
        return parse_openai_answer(stream, body)
    if wire == 'anthropic':
        return parse_anthropic_answer(stream, body)
    return parse_gemini_answer(stream, body)


# ------------------------------------------------------------------ answers rendered in the client's wire

OPENAI_FINISH = {0: 'stop', 1: 'length', 2: 'tool_calls', 3: 'content_filter'}
ANTHROPIC_STOP = {0: 'end_turn', 1: 'max_tokens', 2: 'tool_use', 3: 'refusal'}


def sse_event(event: str | None, data: object) -> str:
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {dump(data)}\n\n'


def render_openai(answer: Answer, model: str, stream: bool) -> str:
    """The answer as a Chat Completions response naming the model actually dialled."""
    usage = {
        'prompt_tokens': answer.input_tokens,
        'completion_tokens': answer.output_tokens,
        'total_tokens': answer.input_tokens + answer.output_tokens,
    }
    finish = OPENAI_FINISH[answer.family]
    if not stream:
        return dump(
            {
                'id': 'chatcmpl-sim',
                'object': 'chat.completion',
                'created': FIXED_OPENAI_CREATED,
                'model': model,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': answer.text}, 'finish_reason': finish}],
                'usage': usage,
            }
        )
    chunk = {'id': 'chatcmpl-sim', 'object': 'chat.completion.chunk', 'created': FIXED_OPENAI_CREATED, 'model': model}
    frames = [
        sse_event(None, {**chunk, 'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': ''}, 'finish_reason': None}]}),
        sse_event(None, {**chunk, 'choices': [{'index': 0, 'delta': {'content': answer.text}, 'finish_reason': None}]}),
        sse_event(None, {**chunk, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': finish}]}),
        sse_event(None, {**chunk, 'choices': [], 'usage': usage}),
        'data: [DONE]\n\n',
    ]
    return ''.join(frames)


def render_anthropic(answer: Answer, model: str, stream: bool) -> str:
    """The answer as a Messages response naming the model actually dialled."""
    stop = ANTHROPIC_STOP[answer.family]
    if not stream:
        return dump(
            {
                'id': 'msg_sim',
                'type': 'message',
                'role': 'assistant',
                'model': model,
                'content': [{'type': 'text', 'text': answer.text}],
                'stop_reason': stop,
                'stop_sequence': None,
                'usage': {'input_tokens': answer.input_tokens, 'output_tokens': answer.output_tokens},
            }
        )
    frames = [
        sse_event(
            'message_start',
            {
                'type': 'message_start',
                'message': {
                    'id': 'msg_sim',
                    'type': 'message',
                    'role': 'assistant',
                    'model': model,
                    'content': [],
                    'stop_reason': None,
                    'stop_sequence': None,
                    'usage': {'input_tokens': answer.input_tokens, 'output_tokens': 0},
                },
            },
        ),
        sse_event('content_block_start', {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}),
        sse_event(
            'content_block_delta',
            {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': answer.text}},
        ),
        sse_event('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
        sse_event(
            'message_delta',
            {'type': 'message_delta', 'delta': {'stop_reason': stop, 'stop_sequence': None}, 'usage': {'output_tokens': answer.output_tokens}},
        ),
        sse_event('message_stop', {'type': 'message_stop'}),
    ]
    return ''.join(frames)


# ------------------------------------------------------------------ dialling one provider


def dial(provider: Json, model_id: str, client_api: str, client_body: Json, client_version: str | None) -> Reply:
    """POST one client request to one provider, in the provider's wire, naming `model_id`.

    The request is carried across in the provider's wire with the echoed fields intact; a
    same-wire request passes through with only its model replaced.
    """
    base = join_base(provider['base'])
    stream = client_body.get('stream') is True
    headers = {'content-type': 'application/json', **auth_headers(provider)}
    if provider['wire'] == 'openai':
        if client_api == 'openai':
            body = {**client_body, 'model': model_id}
            if stream:
                options = body.get('stream_options') if is_record(body.get('stream_options')) else {}
                body['stream_options'] = {**options, 'include_usage': True}
        else:
            body = anthropic_to_openai(client_body, model_id)
        url = f'{base}/v1/chat/completions'
    elif provider['wire'] == 'anthropic':
        headers['anthropic-version'] = client_version if client_api == 'anthropic' and client_version else ANTHROPIC_VERSION
        body = {**client_body, 'model': model_id} if client_api == 'anthropic' else openai_to_anthropic(client_body, model_id)
        url = f'{base}/v1/messages'
    else:  # gemini
        openai_body = client_body if client_api == 'openai' else anthropic_to_openai(client_body, model_id)
        body = openai_to_gemini(openai_body, model_id)
        verb = 'streamGenerateContent' if stream else 'generateContent'
        url = f'{base}/v1beta/models/{quote(model_id, safe="")}:{verb}'
    if stream:
        headers['accept'] = 'text/event-stream'
    return http_call('POST', url, headers, dump(body).encode('utf-8'), DIAL_TIMEOUT_SEC)


class Served:
    """The outcome of serving one request: a provider's parsed answer, or the refusal that ended it."""

    def __init__(self) -> None:
        self.answer: Answer | None = None
        self.model_id = ''
        self.last_status = 502
        self.last_text = ''


def serve(client_api: str, client_body: Json, client_version: str | None) -> Served:
    """Dial the best free model for a request, failing over across refusals.

    Candidates are tried best first, each provider at most once; every refusal is remembered (for
    the demotion ordering) and the first complete answer wins — exactly one successful provider
    call may answer a request, so nothing is dialled in parallel and nothing after a success.
    """
    result = Served()
    requires = request_requirements(client_api, client_body)
    json_on_the_wire = client_api == 'openai' and body_requirements('openai', client_body)['json'] is True
    providers_by_id = {p['id']: p for p in PROVIDERS}
    tried_providers: set[str] = set()
    attempts = 0
    for model in candidates(requires, json_on_the_wire):
        provider = providers_by_id.get(model['provider'])
        if provider is None or provider['id'] in tried_providers:
            continue
        if attempts >= MAX_ATTEMPTS_PER_REQUEST:
            break
        tried_providers.add(provider['id'])
        attempts += 1
        try:
            reply = dial(provider, model['id'], client_api, client_body, client_version)
        except Exception as err:  # the provider could not be reached at all: no status to report
            result.last_status = 502
            result.last_text = str(err)
            note_refusal(provider['id'])
            continue
        if reply.ok:
            answer = parse_answer(provider['wire'], client_body.get('stream') is True, reply.body)
            if answer is not None:
                note_success(provider['id'])
                result.answer = answer
                result.model_id = model['id']
                return result
            result.last_status = 502
            result.last_text = 'the provider answered a body that does not parse in its wire'
            note_refusal(provider['id'])
            continue
        result.last_status = reply.status
        result.last_text = reply.text()
        note_refusal(provider['id'])
    return result


# ------------------------------------------------------------------ the HTTP server


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'makefun-aggregator/1'
    sys_version = ''

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    # -- plumbing -------------------------------------------------------------

    def dispatch(self) -> None:
        self.headers_sent = False
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.completions('openai')
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.completions('anthropic')
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    def send_body(self, status: int, content_type: str, raw: bytes) -> None:
        self.send_response(status)
        self.send_header('content-type', content_type)
        self.send_header('content-length', str(len(raw)))
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_json(self, status: int, body: object) -> None:
        self.send_body(status, 'application/json', dump(body).encode('utf-8'))

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type))

    def send_stream(self, text: str) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        # http.server sends no chunked body without a length; the close ends the stream for the client.
        self.send_header('connection', 'close')
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()
        self.close_connection = True

    def read_json(self) -> Json | None:
        """One client request body, or None when it is not one JSON object or is too large."""
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    # -- the routes -------------------------------------------------------------

    def models(self) -> None:
        """GET /v1/models in the wire the request's auth header names: x-api-key is Anthropic's."""
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list())
        else:
            self.send_json(200, openai_model_list())

    def completions(self, wire: str) -> None:
        """POST a completion on either client wire: dial the providers, render the answer back."""
        body = self.read_json()
        if body is None:
            self.send_error_body(wire, 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        version = self.headers.get('anthropic-version')
        served = serve(wire, body, version)
        if served.answer is None:
            status = served.last_status if 400 <= served.last_status < 600 else 502
            suffix = '' if served.last_text == '' else f': {served.last_text[:400]}'
            error_type = 'api_error' if wire == 'anthropic' else 'upstream_error'
            self.send_error_body(wire, status, error_type, f'every free model refused the request (last HTTP {served.last_status}){suffix}')
            return
        stream = body.get('stream') is True
        if wire == 'openai':
            text = render_openai(served.answer, served.model_id, stream)
        else:
            text = render_anthropic(served.answer, served.model_id, stream)
        if stream:
            self.send_stream(text)
        else:
            self.send_body(200, 'application/json', text.encode('utf-8'))


# ------------------------------------------------------------------ register, poll, start


def register(port: int) -> None:
    """POST /register with the port and the catalog text, embedded verbatim."""
    body = f'{{"protocol_version":{PROTOCOL_VERSION},"port":{port},"catalog":{CATALOG_TEXT}}}'
    reply = http_call(
        'POST', f'{ENV_URL}/register', {'content-type': 'application/json'}, body.encode('utf-8'), PROBE_TIMEOUT_SEC
    )
    if not reply.ok:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {reply.status}: {reply.text()[:500]}')
    parsed = reply.json()
    if parsed is not None and is_number(parsed.get('requests')):
        log(f'registered: the trial holds {parsed["requests"]} requests')


def poll() -> None:
    """Poll GET /work; exit 0 the moment it says done. A server still running at the wall scores 0."""
    while True:
        time.sleep(POLL_INTERVAL_SEC)
        try:
            reply = http_call('GET', f'{ENV_URL}/work', {}, None, 2)
            if reply.status != 200:
                continue
            work = reply.json()
        except Exception:
            continue
        if work is not None and work.get('done') is True:
            sys.stderr.flush()
            os._exit(0)


def stop(*_: object) -> None:
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    load()
    if '--list' in sys.argv[1:]:
        # The same document the register body carries, byte for byte: print the one built object
        # and stop before any server starts. Wait for the flush, so a piped stdout loses nothing.
        sys.stdout.write(CATALOG_TEXT + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns on one layout: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(CATALOG["providers"])} providers and {len(SERVED_MODELS)} free models')
        register(port)
        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
