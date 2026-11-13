"""Official browser OAuth, with callback query logging suppressed.

Credentials are managed by the official SDK in ~/.kaggle, never in this repo.
"""
import logging

from kagglesdk.kaggle_oauth import KaggleOAuth

# BaseHTTPRequestHandler otherwise logs the authorization code in the URL.
KaggleOAuth.OAuthCallbackHandler.log_message = lambda *args, **kwargs: None
logging.getLogger("kagglesdk.kaggle_oauth").setLevel(logging.WARNING)

if __name__ == "__main__":
    from kaggle.api.kaggle_api_extended import KaggleApi

    KaggleApi().auth_login_cli()
