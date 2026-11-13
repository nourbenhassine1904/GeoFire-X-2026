"""Offline security/SDK contract tests; tiny payload fixtures, no real ZIP reads."""
import argparse
import datetime as dt
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import MagicMock,patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from kaggle_blob_session import BlobSessionError,session_metadata,wire_response_metadata,validate_session
from kagglesdk.blobs.types.blob_api_service import ApiStartBlobUploadResponse
import publish_kaggle_single_attempt as publisher

URL='https://storage.googleapis.com/example-bucket/object?signature=SIGNED_QUERY_SENTINEL'
TOKEN='OPAQUE_TOKEN_SENTINEL'


class BlobSessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.network=patch('requests.sessions.Session.send',side_effect=AssertionError('Offline test: HTTP forbidden'))
        cls.network.start()

    @classmethod
    def tearDownClass(cls):cls.network.stop()

    def reject(self,response,code):
        with self.assertRaises(BlobSessionError) as caught:validate_session(response)
        self.assertEqual(caught.exception.code,code)
        text=str(caught.exception)+json.dumps(caught.exception.metadata)
        self.assertNotIn(TOKEN,text);self.assertNotIn('SIGNED_QUERY_SENTINEL',text);self.assertNotIn(URL,text)

    def test_official_camel_wire_decodes_to_snake_attribute(self):
        obj=ApiStartBlobUploadResponse.from_dict(dict(createUrl=URL,token=TOKEN))
        self.assertEqual(obj.create_url,URL);self.assertEqual(obj.createUrl,URL)
        self.assertEqual(obj.token,TOKEN)
        session=validate_session(obj);self.assertEqual(session.url,URL);self.assertEqual(session.token,TOKEN)

    def test_official_roundtrip_preserves_documented_wire_names(self):
        obj=ApiStartBlobUploadResponse.from_dict(dict(createUrl=URL,token=TOKEN))
        self.assertEqual(set(obj.to_dict()),{'createUrl','token'})

    def test_camel_wire_mapping_is_accepted(self):
        self.assertEqual(validate_session(dict(createUrl=URL,token=TOKEN)).url,URL)

    def test_typed_snake_object_is_accepted(self):
        self.assertEqual(validate_session(N(create_url=URL,token=TOKEN)).token,TOKEN)

    def test_object_with_official_camel_alias_is_accepted(self):
        self.assertEqual(validate_session(N(createUrl=URL,token=TOKEN)).url,URL)

    def test_snake_json_is_not_silently_repaired(self):
        self.reject(dict(create_url=URL,token=TOKEN),'BLOB_WIRE_CASE_UNSUPPORTED')
        obj=ApiStartBlobUploadResponse.from_dict(dict(create_url=URL,token=TOKEN))
        self.assertEqual(obj.create_url,'')  # Actual installed SDK ignores non-contract key.
        self.reject(obj,'BLOB_URL_MISSING')

    def test_conflicting_aliases_are_rejected(self):
        self.reject(dict(createUrl=URL,create_url='https://other.example/object',token=TOKEN),'BLOB_URL_ALIAS_CONFLICT')
        self.reject(N(create_url=URL,createUrl='https://other.example/object',token=TOKEN),'BLOB_URL_ALIAS_CONFLICT')

    def test_missing_response_and_nested_envelope_are_rejected(self):
        self.reject(None,'BLOB_RESPONSE_MISSING')
        self.reject(dict(result=dict(createUrl=URL,token=TOKEN)),'BLOB_URL_MISSING')

    def test_missing_url_is_explicit(self):
        for value in (None,''):self.reject(dict(createUrl=value,token=TOKEN),'BLOB_URL_MISSING')

    def test_url_type_and_empty_are_explicit(self):
        for value in (42,[],{}):self.reject(dict(createUrl=value,token=TOKEN),'BLOB_URL_TYPE_INVALID')
        self.reject(dict(createUrl='   ',token=TOKEN),'BLOB_URL_EMPTY')

    def test_missing_token_is_explicit(self):
        for value in (None,''):self.reject(dict(createUrl=URL,token=value),'BLOB_TOKEN_MISSING')

    def test_token_type_and_empty_are_explicit(self):
        for value in (42,[],{}):self.reject(dict(createUrl=URL,token=value),'BLOB_TOKEN_TYPE_INVALID')
        self.reject(dict(createUrl=URL,token='  '),'BLOB_TOKEN_EMPTY')

    def test_non_https_schemes_are_rejected(self):
        for prefix in ('http://','ftp://','file://','//'):
            self.reject(dict(createUrl=prefix+'storage.googleapis.com/object',token=TOKEN),'BLOB_URL_SCHEME_REJECTED')

    def test_valid_bucket_hosts_and_https_port_are_accepted(self):
        for host in ('storage.googleapis.com','bucket.storage.googleapis.com','bucket.with.dots.storage.googleapis.com','storage.googleapis.com:443'):
            self.assertTrue(validate_session(dict(createUrl='https://'+host+'/object',token=TOKEN)))

    def test_other_ports_are_rejected(self):
        self.reject(dict(createUrl='https://storage.googleapis.com:444/object',token=TOKEN),'BLOB_URL_PORT_REJECTED')
        self.reject(dict(createUrl='https://storage.googleapis.com:notaport/object',token=TOKEN),'BLOB_URL_PARSE_FAILED')

    def test_google_json_upload_endpoint_is_accepted_unchanged(self):
        for authority in ('www.googleapis.com','www.googleapis.com:443'):
            url='https://'+authority+'/upload/storage/v1/b/example-bucket/o?upload_id=SIGNED_QUERY_SENTINEL'
            session=validate_session(dict(createUrl=url,token=TOKEN))
            self.assertEqual(session.url,url)
            self.assertEqual(session.token,TOKEN)
            self.assertNotIn('SIGNED_QUERY_SENTINEL',json.dumps(session.metadata))

    def test_google_json_upload_endpoint_rejects_domain_and_path_tricks(self):
        path='/upload/storage/v1/b/example-bucket/o'
        for host in ('www.googleapis.com.evil.example','evilwww.googleapis.com','www.googleapis.com.'):
            self.reject(dict(createUrl='https://'+host+path,token=TOKEN),'BLOB_URL_HOST_REJECTED')
        for invalid in ('/storage/v1/b/example-bucket/o','/upload/storage/v1/b/example-bucket/o/extra',
                        '/upload/storage/v1/b/../o','/upload/storage/v1/b/example%2fbucket/o',
                        '/upload/storage/v1/b/example-bucket/o/../o','//upload/storage/v1/b/example-bucket/o'):
            self.reject(dict(createUrl='https://www.googleapis.com'+invalid,token=TOKEN),'BLOB_URL_PATH_REJECTED')
        self.reject(dict(createUrl='http://www.googleapis.com'+path,token=TOKEN),'BLOB_URL_SCHEME_REJECTED')
        self.reject(dict(createUrl='https://www.googleapis.com:444'+path,token=TOKEN),'BLOB_URL_PORT_REJECTED')
        self.reject(dict(createUrl='https://user@www.googleapis.com'+path,token=TOKEN),'BLOB_URL_USERINFO_REJECTED')
        self.reject(dict(createUrl='https://www.googleapis.com'+path+'#fragment',token=TOKEN),'BLOB_URL_FRAGMENT_REJECTED')

    def test_domain_suffix_tricks_and_unproven_domains_are_rejected(self):
        for host in ('storage.googleapis.com.evil.example','evilgoogleapis.com','www.kaggle.com',
                     'api.kaggle.com','storage.cloud.google.com','127.0.0.1','localhost','storage.googleapis.com.'):
            self.reject(dict(createUrl='https://'+host+'/object',token=TOKEN),'BLOB_URL_HOST_REJECTED')

    def test_empty_or_invalid_bucket_labels_are_rejected(self):
        for host in ('.storage.googleapis.com','a..storage.googleapis.com','-bad.storage.googleapis.com'):
            self.reject(dict(createUrl='https://'+host+'/object',token=TOKEN),'BLOB_URL_HOST_REJECTED')

    def test_userinfo_is_rejected_without_logging_it(self):
        self.reject(dict(createUrl='https://SIGNED_QUERY_SENTINEL:password@storage.googleapis.com/object',token=TOKEN),'BLOB_URL_USERINFO_REJECTED')

    def test_fragment_is_rejected_without_logging_it(self):
        self.reject(dict(createUrl=URL+'#SIGNED_QUERY_SENTINEL',token=TOKEN),'BLOB_URL_FRAGMENT_REJECTED')

    def test_controls_and_backslashes_are_rejected(self):
        for url in (URL+'\n',URL+' ',URL+'\x00',r'https://storage.googleapis.com\@evil.example/object'):
            self.reject(dict(createUrl=url,token=TOKEN),'BLOB_URL_SYNTAX_INVALID')

    def test_ipv6_parse_and_missing_hosts_have_safe_codes(self):
        self.reject(dict(createUrl='https://[invalid/object',token=TOKEN),'BLOB_URL_PARSE_FAILED')
        self.reject(dict(createUrl='https:///object',token=TOKEN),'BLOB_URL_HOST_MISSING')

    def test_metadata_has_no_path_query_userinfo_or_token(self):
        meta=session_metadata(dict(createUrl=URL,token=TOKEN))
        encoded=json.dumps(meta)
        for forbidden in (URL,TOKEN,'SIGNED_QUERY_SENTINEL','example-bucket','/object','signature='):
            self.assertNotIn(forbidden,encoded)
        self.assertEqual(meta['url_host'],'storage.googleapis.com')
        self.assertTrue(meta['token_nonempty'])

    def test_unknown_host_metadata_is_safe_but_does_not_authorize_it(self):
        response=dict(createUrl='https://unexpected.example/private?signature=SIGNED_QUERY_SENTINEL',token=TOKEN)
        self.assertEqual(session_metadata(response)['url_host'],'unexpected.example')
        self.reject(response,'BLOB_URL_HOST_REJECTED')

    def test_token_embedded_in_hostname_is_redacted(self):
        meta=session_metadata(dict(createUrl='https://'+TOKEN.lower()+'.example/object',token=TOKEN))
        self.assertEqual(meta['url_host'],'missing_or_redacted_or_invalid')

    def test_session_repr_is_redacted(self):
        value=validate_session(dict(createUrl=URL,token=TOKEN))
        self.assertEqual(str(value),'BlobSession(<redacted>)')

    def test_wire_projection_does_not_retain_secrets(self):
        meta=wire_response_metadata(json.dumps(dict(createUrl=URL,token=TOKEN)).encode())
        self.assertEqual(meta['url_host'],'storage.googleapis.com')
        for secret in (TOKEN,'SIGNED_QUERY_SENTINEL',URL):self.assertNotIn(secret,json.dumps(meta))

    def test_wire_projection_identifies_case_without_guessing_a_repair(self):
        meta=wire_response_metadata(json.dumps(dict(create_url=URL,token=TOKEN)))
        self.assertFalse(meta['createUrl_present']);self.assertTrue(meta['wire_snake_only'])

    def test_wire_projection_rejects_invalid_json_without_echoing_it(self):
        for raw in ('not JSON SIGNED_QUERY_SENTINEL',b'\xff',None):
            with self.assertRaises(BlobSessionError) as caught:wire_response_metadata(raw)
            self.assertEqual(caught.exception.code,'BLOB_WIRE_JSON_INVALID')
            self.assertNotIn('SIGNED_QUERY_SENTINEL',str(caught.exception))

    def test_wire_projection_rejects_non_object_and_oversized_bodies(self):
        with self.assertRaises(BlobSessionError) as caught:wire_response_metadata('[]')
        self.assertEqual(caught.exception.code,'BLOB_WIRE_SHAPE_INVALID')
        with self.assertRaises(BlobSessionError) as caught:wire_response_metadata(b'x'*65_537)
        self.assertEqual(caught.exception.code,'BLOB_WIRE_RESPONSE_TOO_LARGE')

    def test_encoded_and_non_ascii_host_tricks_are_rejected(self):
        for host in ('storage.googleapis.com%2eevil.example','st\u043erage.googleapis.com'):
            self.reject(dict(createUrl='https://'+host+'/object',token=TOKEN),'BLOB_URL_HOST_INVALID')

    def test_security_rejection_prevents_put_and_dataset_creation(self):
        self._publisher_simulation('https://unexpected.example/object?signature=SIGNED_QUERY_SENTINEL',TOKEN,'BLOB_URL_HOST_REJECTED')

    def test_missing_token_prevents_put_and_dataset_creation(self):
        self._publisher_simulation(URL,'','BLOB_TOKEN_MISSING')

    def test_validated_session_permits_only_explicit_private_create(self):
        self._publisher_simulation(URL,TOKEN,None)

    def _publisher_simulation(self,url,token,expected_code):
        # Entire publication path is simulated. No actual ZIP or account data is read.
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);payload=root/'tiny.bin';payload.write_bytes(b'fixture')
            credentials_path=root/'credentials-fixture.json';credentials_path.write_text('{}')
            receipt=dict(handle='nnnnbhss/geofirex-wsts-fold0-tv-001-v1',payload_bytes=4829918684,
                sha256='464e89146fa6fdabfccdd1eb81d97b08549ca9b14428a32e3993f5996e8f5890',
                lot_identity='29ccf1eaa0f4d1fb0ab4333ee307e8af881f4255d1bd5639721ccaea2c540f81')
            create=MagicMock(return_value=N(error='',status='ok'))
            client=N(blobs=N(blob_api_client=N(start_blob_upload=MagicMock(return_value=N(create_url=url,token=token)))),
                     datasets=N(dataset_api_client=N(create_dataset=create)))
            context=MagicMock();context.__enter__.return_value=client
            creds=N(_access_token='OFFLINE_DUMMY',_access_token_expiration=dt.datetime(2099,1,1,tzinfo=dt.timezone.utc),
                    introspect=lambda:'nnnnbhss')
            args=argparse.Namespace(staged='not-read',execute=True,credentials=str(credentials_path),attempt=str(root/'attempt'),
                journal=str(root/'budget.jsonl'),max_http_body_bytes=6_000_000_000,stop_at_cumulative_body_bytes=6_000_000_000,license='CC-BY-4.0')
            with patch.object(publisher,'check_staged',return_value=(payload,receipt)), \
                 patch.object(publisher,'collect_dataset_preflight',return_value=dict(decision=dict(allow_create_new=True))), \
                 patch('kagglesdk.KaggleClient',return_value=context), \
                 patch('kagglesdk.kaggle_creds.KaggleCredentials.load',return_value=creds), \
                 patch.object(publisher,'stream_once') as put:
                result=publisher.publish(args)
            if expected_code:
                self.assertEqual(result['local_error_code'],expected_code)
                self.assertEqual(result['error_stage'],'validate_blob_session')
                put.assert_not_called();create.assert_not_called()
            else:
                self.assertEqual(result['status'],'PRIVATE_CREATE_ACCEPTED_CERTIFICATION_PENDING')
                put.assert_called_once();create.assert_called_once()
                request=create.call_args.args[0]
                self.assertTrue(request.is_private);self.assertEqual(request.owner_slug,'nnnnbhss')
                self.assertEqual(request.slug,'geofirex-wsts-fold0-tv-001-v1');self.assertEqual(request.files[0].token,TOKEN)
            for file in (root/'attempt').glob('*.json'):
                text=file.read_text()
                self.assertNotIn('SIGNED_QUERY_SENTINEL',text);self.assertNotIn(TOKEN,text)


if __name__=='__main__':unittest.main()
