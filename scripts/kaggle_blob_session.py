"""Offline validation and secret-free diagnostics for official StartBlobUpload.

Wire contract: createUrl/token. Official SDK object: create_url/token, with a
read-only createUrl alias. Google JSON API uploads use an exact host/path rule.
"""
from collections.abc import Mapping
from dataclasses import dataclass,field
import json
import re
import urllib.parse


class BlobSessionError(RuntimeError):
    def __init__(self,code,metadata=None):
        # Never include the raw object, signed URL, token, or parser exception.
        super().__init__(code)
        self.code=code
        self.metadata=metadata or {}


@dataclass(frozen=True,repr=False)
class BlobSession:
    url:str=field(repr=False)
    token:str=field(repr=False)
    metadata:dict=field(repr=False)

    def __repr__(self):return 'BlobSession(<redacted>)'


def _kind(value):
    if value is None:return 'null'
    if isinstance(value,str):return 'string'
    return 'non_string'


def _present(value):return isinstance(value,str) and bool(value.strip())


def _fields(response):
    if isinstance(response,Mapping):
        camel=response.get('createUrl');snake=response.get('create_url');token=response.get('token')
        return camel if 'createUrl' in response else snake,token,dict(
            response_form='mapping',createUrl_present='createUrl' in response,create_url_present='create_url' in response,
            token_field_present='token' in response,alias_conflict='createUrl' in response and 'create_url' in response and camel!=snake,
            wire_snake_only='create_url' in response and 'createUrl' not in response)
    if response is None:return None,None,dict(response_form='missing',alias_conflict=False,wire_snake_only=False)
    try:
        snake=getattr(response,'create_url',None);camel=getattr(response,'createUrl',None);token=getattr(response,'token',None)
    except Exception:
        raise BlobSessionError('BLOB_RESPONSE_FIELD_ACCESS_FAILED') from None
    return snake if snake is not None else camel,token,dict(
        response_form='object',createUrl_present=camel is not None,create_url_present=snake is not None,
        token_field_present=hasattr(response,'token'),alias_conflict=snake is not None and camel is not None and snake!=camel,
        wire_snake_only=False)


def session_metadata(response):
    """Only fixed flags and DNS hostname; never path, query, token, lengths or repr."""
    try:url,token,meta=_fields(response)
    except BlobSessionError as error:return dict(response_form='unreadable',metadata_error=error.code)
    meta.update(url_type=_kind(url),url_nonempty=_present(url),token_type=_kind(token),token_nonempty=_present(token))
    if not _present(url):return meta
    try:
        parsed=urllib.parse.urlsplit(url)
        host=parsed.hostname
        meta['url_scheme']=parsed.scheme if parsed.scheme in ('https','http') else 'other_or_missing'
        meta['url_has_userinfo']=parsed.username is not None or parsed.password is not None
        meta['url_has_fragment']=bool(parsed.fragment)
        meta['url_has_backslash']='\\' in url
        meta['url_has_whitespace']=any(c.isspace() or ord(c)<32 or ord(c)==127 for c in url)
        # Hostnames are safe metadata only after syntax and known-token checks.
        safe=host and len(host)<=253 and re.fullmatch(r'[a-zA-Z0-9.-]+',host)
        if safe and not (_present(token) and token.lower() in host.lower()):meta['url_host']=host.lower()
        else:meta['url_host']='missing_or_redacted_or_invalid'
        try:meta['url_port']=parsed.port
        except ValueError:meta['url_port']='invalid'
    except (ValueError,TypeError):meta['url_parse_error']=True
    return meta


def wire_response_metadata(raw):
    """Project a bounded raw HTTP body without retaining its secret-bearing text."""
    if not isinstance(raw,(str,bytes,bytearray)):
        raise BlobSessionError('BLOB_WIRE_JSON_INVALID')
    size=len(raw.encode('utf-8')) if isinstance(raw,str) else len(raw)
    if size>65_536:raise BlobSessionError('BLOB_WIRE_RESPONSE_TOO_LARGE')
    try:value=json.loads(raw)
    except (ValueError,TypeError,UnicodeError):raise BlobSessionError('BLOB_WIRE_JSON_INVALID') from None
    if not isinstance(value,Mapping):raise BlobSessionError('BLOB_WIRE_SHAPE_INVALID')
    return session_metadata(value)


def _allowed_gcs_host(host):
    if host=='storage.googleapis.com':return True
    if not host or not host.endswith('.storage.googleapis.com'):return False
    prefix=host[:-len('.storage.googleapis.com')]
    # Strict DNS labels; prevents suffix tricks, empty labels and malformed names.
    return bool(prefix) and all(re.fullmatch(r'[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?',label) for label in prefix.split('.'))


def validate_session(response):
    meta=session_metadata(response)
    def fail(code):raise BlobSessionError(code,meta)
    url,token,flags=_fields(response)
    if flags['response_form']=='missing':fail('BLOB_RESPONSE_MISSING')
    if flags['alias_conflict']:fail('BLOB_URL_ALIAS_CONFLICT')
    # Snake-case JSON is not the official wire contract. Do not silently repair it.
    if flags['wire_snake_only']:fail('BLOB_WIRE_CASE_UNSUPPORTED')
    if url is None or url=='':fail('BLOB_URL_MISSING')
    if not isinstance(url,str):fail('BLOB_URL_TYPE_INVALID')
    if not url.strip():fail('BLOB_URL_EMPTY')
    if any(c.isspace() or ord(c)<32 or ord(c)==127 for c in url) or '\\' in url:
        fail('BLOB_URL_SYNTAX_INVALID')
    try:
        parsed=urllib.parse.urlsplit(url);host=parsed.hostname;port=parsed.port
    except (ValueError,TypeError):fail('BLOB_URL_PARSE_FAILED')
    if parsed.scheme!='https':fail('BLOB_URL_SCHEME_REJECTED')
    if parsed.username is not None or parsed.password is not None:fail('BLOB_URL_USERINFO_REJECTED')
    if parsed.fragment:fail('BLOB_URL_FRAGMENT_REJECTED')
    if not host:fail('BLOB_URL_HOST_MISSING')
    if not re.fullmatch(r'[a-zA-Z0-9.-]+',host) or len(host)>253:fail('BLOB_URL_HOST_INVALID')
    if host.lower()=='www.googleapis.com':
        # Official GCS JSON object upload endpoint. Match the original raw path:
        # no decoding, rewriting, extra segments or arbitrary Google API paths.
        # https://docs.cloud.google.com/storage/docs/request-endpoints
        if not re.fullmatch(r'/upload/storage/v1/b/[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]/o',parsed.path):
            fail('BLOB_URL_PATH_REJECTED')
    elif not _allowed_gcs_host(host.lower()):fail('BLOB_URL_HOST_REJECTED')
    # An omitted HTTPS port is the standard port 443.
    if port not in (None,443):fail('BLOB_URL_PORT_REJECTED')
    if token is None or token=='':fail('BLOB_TOKEN_MISSING')
    if not isinstance(token,str):fail('BLOB_TOKEN_TYPE_INVALID')
    if not token.strip():fail('BLOB_TOKEN_EMPTY')
    return BlobSession(url,token,meta)
