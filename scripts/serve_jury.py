"""Launch the local jury app; no training, recovery, Spark or publication."""
import argparse
import sys
from geofirex.jury_app import create_jury_app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8014)
    args=parser.parse_args()
    import uvicorn
    # Proactor can lose its listening socket on WinError 64 during rapid navigation.
    loop='asyncio:SelectorEventLoop' if sys.platform=='win32' else 'auto'
    uvicorn.run(create_jury_app(),host='127.0.0.1',port=args.port,access_log=False,loop=loop)
