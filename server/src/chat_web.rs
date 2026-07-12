use axum::response::Html;
use axum::routing::get;
use axum::Router;

pub fn routes() -> Router {
    Router::new().route("/chat", get(chat_page))
}

async fn chat_page() -> Html<&'static str> {
    Html(CHAT_HTML)
}

const CHAT_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Chat</title>
<style>
  *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
  body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:#212121;color:#e0e0e0;height:100vh;display:flex;flex-direction:column}
  header{background:#171717;padding:14px 24px;border-bottom:1px solid #333;font-size:18px;font-weight:600;display:flex;align-items:center;gap:10px}
  header span{color:#10a37f}
  #messages{flex:1;overflow-y:auto;padding:24px 16px;display:flex;flex-direction:column;gap:16px;scroll-behavior:smooth}
  .msg{max-width:720px;width:fit-content;padding:12px 16px;border-radius:10px;line-height:1.55;font-size:15px;white-space:pre-wrap;word-break:break-word}
  .msg.user{background:#2f2f2f;align-self:flex-end;border-bottom-right-radius:4px}
  .msg.assistant{background:#3a3a3a;align-self:flex-start;border-bottom-left-radius:4px}
  .msg.system{display:none}
  .label{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:.5px;color:#888;margin-bottom:4px;display:block}
  .thinking{display:flex;align-items:center;gap:6px;padding:12px 16px;color:#888;font-size:14px}
  .thinking .dot{width:8px;height:8px;background:#10a37f;border-radius:50%;animation:bounce 1.4s infinite ease-in-out both}
  .thinking .dot:nth-child(1){animation-delay:-.32s}
  .thinking .dot:nth-child(2){animation-delay:-.16s}
  @keyframes bounce{0%,80%,100%{transform:scale(0)}40%{transform:scale(1)}}
  #input-area{background:#171717;border-top:1px solid #333;padding:16px 24px}
  #input-row{max-width:720px;margin:0 auto;display:flex;gap:8px}
  #input-box{flex:1;background:#2f2f2f;border:1px solid #444;border-radius:8px;padding:12px 16px;color:#e0e0e0;font-size:15px;outline:none;resize:none;font-family:inherit;line-height:1.4;max-height:120px}
  #input-box:focus{border-color:#10a37f}
  #send-btn{background:#10a37f;color:#fff;border:none;border-radius:8px;width:48px;height:48px;cursor:pointer;display:flex;align-items:center;justify-content:center;flex-shrink:0;transition:background .15s}
  #send-btn:hover{background:#0e8c6b}
  #send-btn:disabled{opacity:.4;cursor:not-allowed}
  #send-btn svg{width:20px;height:20px;fill:currentColor}
  @media(max-width:600px){#messages{padding:16px 12px}#input-area{padding:12px 16px}.msg{font-size:14px}}
</style>
</head>
<body>
<header><span>&#9670;</span> Chat</header>
<div id="messages">
  <div class="msg assistant"><span class="label">assistant</span>Hello! How can I help you today?</div>
</div>
<div id="input-area">
  <div id="input-row">
    <textarea id="input-box" rows="1" placeholder="Send a message..."></textarea>
    <button id="send-btn" disabled aria-label="Send"><svg viewBox="0 0 24 24"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"/></svg></button>
  </div>
</div>
<script>
const messagesEl=document.getElementById('messages');
const inputBox=document.getElementById('input-box');
const sendBtn=document.getElementById('send-btn');
let loading=false;

function autoResize(){inputBox.style.height='auto';inputBox.style.height=Math.min(inputBox.scrollHeight,120)+'px'}
inputBox.addEventListener('input',autoResize);

inputBox.addEventListener('input',()=>{sendBtn.disabled=!inputBox.value.trim()||loading});

function addMessage(role,content){
  const div=document.createElement('div');
  div.className='msg '+role;
  div.innerHTML='<span class="label">'+role+'</span>'+escapeHtml(content);
  messagesEl.appendChild(div);
  messagesEl.scrollTop=messagesEl.scrollHeight;
}

function showThinking(){
  const div=document.createElement('div');
  div.className='thinking';
  div.id='thinking';
  div.innerHTML='<span class="dot"></span><span class="dot"></span><span class="dot"></span>';
  messagesEl.appendChild(div);
  messagesEl.scrollTop=messagesEl.scrollHeight;
}

function hideThinking(){
  const el=document.getElementById('thinking');
  if(el)el.remove();
}

function escapeHtml(s){
  const d=document.createElement('div');
  d.textContent=s;
  return d.innerHTML;
}

async function sendMessage(){
  const text=inputBox.value.trim();
  if(!text||loading)return;
  inputBox.value='';
  autoResize();
  sendBtn.disabled=true;
  addMessage('user',text);
  loading=true;
  showThinking();
  try{
    const hist=[];
    const msgs=messagesEl.querySelectorAll('.msg');
    msgs.forEach(m=>{
      const role=m.classList.contains('user')?'user':'assistant';
      const content=m.textContent;
      if(content&&role!=='system')hist.push({role,content});
    });
    const resp=await fetch('/v1/chat/completions',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({model:'default',messages:hist})
    });
    if(!resp.ok)throw new Error('HTTP '+resp.status);
    const data=await resp.json();
    hideThinking();
    const reply=data.choices[0].message.content;
    addMessage('assistant',reply);
  }catch(e){
    hideThinking();
    addMessage('assistant','Error: '+(e.message||'request failed'));
  }finally{
    loading=false;
    sendBtn.disabled=!inputBox.value.trim();
  }
}

sendBtn.addEventListener('click',sendMessage);
inputBox.addEventListener('keydown',e=>{
  if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();sendMessage()}
});
</script>
</body>
</html>"#;
