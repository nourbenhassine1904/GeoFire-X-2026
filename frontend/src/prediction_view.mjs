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
  const rows=['event_id,as_of,target_date,checkpoint_sha256,threshold,preview_stride,row_pixel,col_pixel,probability,observed_J,predicted_Jplus1,new_predicted_Jplus1,target_Jplus1,new_target_Jplus1,error,error_new,model,source_archive_sha256,saved_member_sha256,scope,full_height,full_width,crs,coordinate_reference,input_dates,provenance'];
  const cell=v=>{const s=String(v??'');return /[",\n\r]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s;};
  result.probability.forEach((row,y)=>row.forEach((p,x)=>{
    rows.push([result.event_id,result.as_of,result.target_date,result.checkpoint_sha256,result.threshold,result.preview_stride,
      (result.tile_origin?.[0]||0)+y*result.preview_stride,(result.tile_origin?.[1]||0)+x*result.preview_stride,p,result.observed_fire[y][x],result.predicted_fire[y][x],result.new_predicted_fire?.[y][x]??'',
      layers?.ground[y][x]??'',layers?.newTarget[y][x]??'',layers?.errors[y][x]??'',layers?.newErrors[y][x]??'',
      result.model,result.source_archive_sha256,result.saved_member_sha256,result.scope,...result.full_shape,result.crs,result.crs?'raster CRS supplied':'pixel grid; no CRS',result.input_dates?.join('|'),result.provenance||'CPU historical interactive inference'].map(cell).join(','));
  }));return rows.join('\n');
}
export function downloadText(content,filename,type){
  const url=URL.createObjectURL(new Blob([content],{type})),a=document.createElement('a');
  a.href=url;a.download=filename;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
}
export async function requestForecast(apiBase,eventId,day,{signal,onPrediction=()=>{},revealTruth=true}={}){
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
  if(!revealTruth)return {result,truth:null,truthError:''};
  // Separate request after inference, never a model input.
  const {response:actual,body:target}=await jsonRequest(`${apiBase}/ground-truth?${query}`);
  if(!actual.ok)return {result,truth:null,truthError:target.detail||'Vérité terrain indisponible'};
  if(JSON.stringify(target.full_shape)!==JSON.stringify(result.full_shape)||target.preview_stride!==result.preview_stride||target.crs!==result.crs||JSON.stringify(target.transform)!==JSON.stringify(result.transform))
    return {result,truth:null,truthError:'Géométrie de vérité terrain incompatible'};
  return {result,truth:target.ground_truth,truthError:''};
}

// The viewport is chosen from information available at J and predictions only.
// Ground truth never changes this crop when revealed.
export function activeBounds(grids){
  const grid=grids.find(g=>g?.length);if(!grid)return null;
  const height=grid.length,width=grid[0].length;let top=height,left=width,bottom=-1,right=-1;
  for(const g of grids.filter(Boolean))g.forEach((row,y)=>row.forEach((v,x)=>{if(v===1){top=Math.min(top,y);left=Math.min(left,x);bottom=Math.max(bottom,y);right=Math.max(right,x);}}));
  if(bottom<0)return {top:0,left:0,height,width};
  const pad=Math.max(6,Math.ceil(Math.max(bottom-top+1,right-left+1)*.18));
  top=Math.max(0,top-pad);left=Math.max(0,left-pad);bottom=Math.min(height-1,bottom+pad);right=Math.min(width-1,right+pad);
  return {top,left,height:bottom-top+1,width:right-left+1};
}
export const cropGrid=(grid,bounds)=>bounds?grid?.slice(bounds.top,bounds.top+bounds.height).map(row=>row.slice(bounds.left,bounds.left+bounds.width)):grid;
export function previewMetrics(result,truth,newOnly=false){
  if(!truth)return null;
  let tp=0,fp=0,fn=0,tn=0,brier=0;const values=[];
  truth.forEach((row,y)=>row.forEach((target,x)=>{
    const p=result.probability[y][x];if(target===null||p===null||(newOnly&&result.observed_fire[y][x]!==0))return;
    const predicted=result.predicted_fire[y][x]===1;
    if(predicted){if(target===1)tp++;else fp++;}else if(target===1)fn++;else tn++;
    brier+=(p-target)**2;values.push([p,target]);
  }));
  values.sort((a,b)=>b[0]-a[0]);let hits=0,previous=0,area=0;
  for(let i=0;i<values.length;i++){hits+=values[i][1];if(i===values.length-1||values[i][0]!==values[i+1][0]){area+=(hits-previous)*hits/(i+1);previous=hits;}}
  const ratio=(a,b)=>b?a/b:null;
  return {tp,fp,fn,tn,valid_pixels:values.length,recall:ratio(tp,tp+fn),precision:ratio(tp,tp+fp),dice:ratio(2*tp,2*tp+fp+fn),AP:tp+fn?area/(tp+fn):values.length?0:null,Brier:ratio(brier,values.length)};
}
