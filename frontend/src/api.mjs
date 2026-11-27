// A 409 means a read/inference has not started. Only that busy response is retried.
export async function readJSON(url,{signal,...options}={}){
 for(let attempt=0;;attempt++){
  const response=await fetch(url,{...options,signal});
  let body;try{body=await response.json();}catch{throw Error('Réponse du service illisible ('+response.status+').');}
  if(response.status!==409||attempt>=11){if(!response.ok)throw Error(body.detail||'Service indisponible ('+response.status+').');return body;}
  await new Promise((resolve,reject)=>{
   const finish=()=>{signal?.removeEventListener('abort',abort);resolve();};
   const timer=setTimeout(finish,Math.min(200*2**attempt,1800));
   const abort=()=>{clearTimeout(timer);signal?.removeEventListener('abort',abort);reject(new DOMException('Aborted','AbortError'));};
   if(signal?.aborted)abort();else signal?.addEventListener('abort',abort,{once:true});
  });
 }
}
