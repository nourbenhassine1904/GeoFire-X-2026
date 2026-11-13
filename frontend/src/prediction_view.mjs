// Shared by the canvas component and the end-to-end pixel checks.
export function imagePixels(grid,kind='binary'){
  const height=grid.length,width=grid[0].length,pixels=new Uint8ClampedArray(height*width*4);
  grid.forEach((row,y)=>row.forEach((value,x)=>{
    const color=value===null?[85,100,108]:kind==='errors'?[[20,39,45],[48,187,128],[242,82,82],[63,142,252]][value]:kind==='probability'
      ?[Math.round(255*value),Math.round(150*value),Math.round(70*(1-value))]:value?[255,184,70]:[20,39,45];
    pixels.set([...color,255],(y*width+x)*4);
  }));return {width,height,pixels};
}
export function comparisonLayers(result,truth){
  if(!truth)return null;
  const ground=truth.map((row,y)=>row.map((v,x)=>result.probability[y][x]===null?null:v));
  const newTarget=ground.map((row,y)=>row.map((v,x)=>v===null?null:Number(v===1&&result.observed_fire[y][x]===0)));
  const errors=ground.map((row,y)=>row.map((v,x)=>v===null?null:result.predicted_fire[y][x]?(v?1:2):(v?3:0)));
  const newErrors=errors.map((row,y)=>row.map((v,x)=>result.observed_fire[y][x]===1?null:v));
  return {ground,newTarget,errors,newErrors};
}
export function forecastCSV(result,truth){
  const layers=comparisonLayers(result,truth);
  const rows=['event_id,as_of,target_date,checkpoint_sha256,threshold,preview_stride,row_pixel,col_pixel,probability,observed_J,predicted_Jplus1,new_predicted_Jplus1,target_Jplus1,new_target_Jplus1,error,error_new'];
  result.probability.forEach((row,y)=>row.forEach((p,x)=>{
    rows.push([result.event_id,result.as_of,result.target_date,result.checkpoint_sha256,result.threshold,result.preview_stride,
      y*result.preview_stride,x*result.preview_stride,p,result.observed_fire[y][x],result.predicted_fire[y][x],result.new_predicted_fire?.[y][x]??'',
      layers?.ground[y][x]??'',layers?.newTarget[y][x]??'',layers?.errors[y][x]??'',layers?.newErrors[y][x]??''].map(v=>v??'').join(','));
  }));return rows.join('\n');
}
export function downloadText(content,filename,type){
  const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');
  a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
export async function requestForecast(apiBase,eventId,day,{signal,onPrediction=()=>{}}={}){
  const query=new URLSearchParams({event_id:eventId,as_of:day,max_side:512});
  const jsonRequest=async url=>{
    for(let attempt=0;;attempt++){
      const response=await fetch(url,{signal}),body=await response.json();
      if(response.status!==409||attempt>=5)return {response,body};
      // A409 means inference has not started. Retry a busy reader, never a completed job.
      await new Promise((resolve,reject)=>{
        const abort=()=>{clearTimeout(timer);reject(new DOMException('Aborted','AbortError'));};
        const timer=setTimeout(()=>{signal?.removeEventListener('abort',abort);resolve();},Math.min(200*2**attempt,1000));
        if(signal?.aborted)abort();else signal?.addEventListener('abort',abort,{once:true});
      });
    }
  };
  const {response,body:result}=await jsonRequest(`${apiBase}/predict?${query}`);
  if(!response.ok)throw new Error(result.detail||'La prédiction a échoué');
  if(signal?.aborted)throw new DOMException('Aborted','AbortError');
  onPrediction(result);
  // Separate request after inference, never a model input.
  const {response:actual,body:target}=await jsonRequest(`${apiBase}/ground-truth?${query}`);
  if(!actual.ok)return {result,truth:null,truthError:target.detail||'Vérité terrain indisponible'};
  if(JSON.stringify(target.full_shape)!==JSON.stringify(result.full_shape)||target.preview_stride!==result.preview_stride||target.crs!==result.crs||JSON.stringify(target.transform)!==JSON.stringify(result.transform))
    return {result,truth:null,truthError:'Géométrie de vérité terrain incompatible'};
  return {result,truth:target.ground_truth,truthError:''};
}
