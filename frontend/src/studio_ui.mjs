import React,{createContext,useContext,useEffect,useState} from 'react';
import Aurora from './reactbits/Aurora.jsx';
import SplitText from './reactbits/SplitText.jsx';
import CountUp from './reactbits/CountUp.jsx';
import SpotlightCard from './reactbits/SpotlightCard.jsx';
import FadeContent from './reactbits/FadeContent.jsx';
export const h=React.createElement;
export const MotionContext=createContext({enabled:false});
export const number=(value,digits=0)=>Number.isFinite(value)?value.toLocaleString('fr-FR',{maximumFractionDigits:digits}):'—';
const paths={
 overview:'M3 3h7v7H3z M14 3h7v7h-7z M3 14h7v7H3z M14 14h7v7h-7z',
 studio:'M3 8V4h4 M17 4h4v4 M21 16v4h-4 M7 20H3v-4 M8 12h8 M12 8v8',
 analytics:'M4 20V10 M10 20V4 M16 20v-8 M22 20H2',
 data:'M3 6c0-4 18-4 18 0s-18 4-18 0 M3 6v6c0 4 18 4 18 0V6 M3 12v6c0 4 18 4 18 0v-6',
 research:'M3 4h6c2 0 3 1 3 2 0-1 1-2 3-2h6v16h-6c-2 0-3 1-3 1s-1-1-3-1H3z M12 6v15',
 arrow:'M5 12h14 M13 6l6 6-6 6',chevron:'m9 5 7 7-7 7',
 lock:'M6 11h12v10H6z M8 11V7a4 4 0 0 1 8 0v4',
 check:'m5 12 4 4L19 6',search:'M16 16l5 5 M18 10a8 8 0 1 1-16 0 8 8 0 0 1 16 0',
 menu:'M4 6h16 M4 12h16 M4 18h16',close:'m6 6 12 12 M18 6 6 18',
 layers:'m12 2 10 6-10 6L2 8z M2 12l10 6 10-6 M2 16l10 6 10-6',
 info:'M12 11v6 M12 7h.01 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
 download:'M12 3v12 m-5-5 5 5 5-5 M4 17v4h16v-4',
 clock:'M12 6v6l4 2 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
 motion:'M3 8c4-8 14 8 18 0 M3 16c4-8 14 8 18 0',
 flame:'M13 2c1 6-4 5-2 10 2-1 3-3 3-3 4 4 6 6 4 10-3 6-13 4-13-2 0-5 5-7 8-15z',
 code:'m8 5-6 7 6 7 M16 5l6 7-6 7 M14 3l-4 18',
 shield:'m12 2 9 4v6c0 5-9 10-9 10S3 17 3 12V6z M8 12l3 3 5-6'
};
export function Icon({name,size=18,...props}){return h('svg',{width:size,height:size,viewBox:'0 0 24 24',fill:'none',stroke:'currentColor',strokeWidth:1.6,strokeLinecap:'round',strokeLinejoin:'round','aria-hidden':true,focusable:false,...props},h('path',{d:paths[name]||paths.layers}));}
export function Brand(){return h('span',{className:'brand-content'},h('span',{className:'brand-mark'},h(Icon,{name:'flame',size:21})),h('span',null,'GEOFIRE',h('span',{className:'brand-x'},'–X')));}
export function useResource(url){
 const [revision,setRevision]=useState(0),[state,setState]=useState({data:null,error:''});
 useEffect(()=>{const c=new AbortController();setState({data:null,error:''});
  fetch(url,{signal:c.signal}).then(async r=>{if(!r.ok)throw new Error(`Source indisponible (${r.status})`);const data=await r.json();if(!c.signal.aborted)setState({data,error:''});}).catch(e=>{if(e.name!=='AbortError')setState({data:null,error:e.message});});
  return()=>c.abort();
 },[url,revision]);
 return {...state,retry:()=>setRevision(r=>r+1)};
}
export function SourceState({state}){return state.error?h('div',{className:'source-error',role:'alert'},h(Icon,{name:'info'}),h('span',null,state.error),h('button',{onClick:state.retry},'Réessayer')):!state.data?h('span',{className:'loading-note',role:'status'},'Lecture des preuves enregistrées…'):null;}
export function Fade({children,className='',trigger='scroll'}){
 const {enabled}=useContext(MotionContext),[state,setState]=useState('waiting');
 return enabled?h(FadeContent,{duration:.48,initialOpacity:.08,threshold:.08,trigger,once:true,className,'data-react-bits':'FadeContent','data-animation-state':state,onStart:()=>setState('running'),onComplete:()=>setState('complete')},children):h('div',{className,'data-motion-fallback':'FadeContent'},children);
}
export function WelcomeTitle(){const {enabled}=useContext(MotionContext);return enabled?h(SplitText,{tag:'h1',text:'Comprendre le feu. Anticiper J+1.',className:'hero-title',splitType:'words',textAlign:'left',delay:85,duration:.8,effect:'rise',trigger:'load'}):h('h1',{className:'hero-title'},'Comprendre le feu. Anticiper J+1.');}
export function AuroraBackdrop(){const {enabled}=useContext(MotionContext);return h('div',{className:'hero-aurora','aria-hidden':true,'data-motion-fallback':enabled?undefined:'Aurora'},enabled&&h(Aurora,{colorStops:['#F97316','#FFB020','#FDBA8C'],lightMode:true,speed:.55,amplitude:.85,brightness:1,blend:.65,height:.68,opacity:.85,dpr:1,mouseInteraction:false,intro:false}));}
export function StatCard({icon,label,value,detail,suffix='',digits=0}){
 const {enabled}=useContext(MotionContext),valid=Number.isFinite(value),[countState,setCountState]=useState('waiting');
 const body=[h('div',{className:'stat-heading',key:'title'},h('span',null,label),h(Icon,{name:icon})),
  h('div',{className:'stat-value',key:'value','aria-label':valid?number(value,digits)+suffix:'Non disponible','data-react-bits':enabled&&valid?'CountUp':undefined,'data-animation-state':enabled?countState:'static'},h('span',{'aria-hidden':true},valid?(enabled?h(CountUp,{to:value,duration:1.35,separator:' ',decimals:digits,onStart:()=>setCountState('running'),onEnd:()=>setCountState('complete')}):number(value,digits)):'—'),h('span',{className:'stat-unit','aria-hidden':true},suffix)),h('p',{key:'detail'},detail)];
 return enabled?h(SpotlightCard,{className:'stat-card',theme:'light',spotlightColor:'#F97316',intensity:.85,borderGlow:.9,spotlightSize:260,proximity:24,flare:false,'data-react-bits':'SpotlightCard'},...body):h('div',{className:'stat-card spotlight-card','data-motion-fallback':'SpotlightCard'},...body);
}
export function PageHeader({kicker,title,description,action}){return h('header',{className:'page-header'},h('div',null,h('div',{className:'eyebrow'},kicker),h('h1',null,title),h('p',null,description)),action);}
export function Badge({children,tone=''}){return h('span',{className:`badge ${tone}`},children);}
export function downloadJSON(data,name){const a=document.createElement('a'),url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'}));a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
