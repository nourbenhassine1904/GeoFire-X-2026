"""Launch the local jury app; no training, recovery, Spark or publication."""
import argparse
from geofirex.jury_app import create_jury_app


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port',type=int,default=8014)
    args=parser.parse_args()
    import uvicorn
    uvicorn.run(create_jury_app(),host='127.0.0.1',port=args.port,access_log=False)
