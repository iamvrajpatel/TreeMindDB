"use client";

import { ChangeEvent, FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Background, Controls, ReactFlow, BaseEdge, getBezierPath, type EdgeProps, type NodeProps } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import { API_BASE_URL, api } from "../lib/api";

type TreeNode = { id: string; parent_id: string | null; label: string; summary: string; depth: number; node_type: string };
type Conversation = { id: string; title: string; active_tokens: number; profile_summary: string };
type TraceEvent = { event: string; payload: any };
type Turn = { role: "user" | "assistant"; text: string; sources?: string; pending?: string[]; traceId?: string };

function NeuralNode({ data }: NodeProps<any>) { return <button onClick={data.onInspect} className={`neural-node ${data.root?"neural-root":""} ${data.focused?"is-focused":""} ${data.dimmed?"is-dimmed":""} ${data.state||""}`}><span className="node-orbit"/><span className="node-core"><b>{data.label}</b><small>{data.meta}</small></span><span className="node-badge">{data.count ?? "◈"}</span></button>; }
function EnergyEdge(props: EdgeProps<any>) { const [path]=getBezierPath(props); const mode=props.data?.mode||"idle"; const delay=`${(props.data?.delay||0)*.18}s`; return <g className={`energy-edge ${mode}`}><BaseEdge path={path} className="energy-base"/><path d={path} className="energy-travel" style={{animationDelay:delay}}/></g>; }
const nodeTypes={neural:NeuralNode}; const edgeTypes={energy:EnergyEdge};

export default function TreeMind() {
  const [nodes, setNodes] = useState<TreeNode[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversation, setConversation] = useState<Conversation>();
  const [selected, setSelected] = useState<any>();
  const [turns, setTurns] = useState<Turn[]>([]);
  const [routeIds, setRouteIds] = useState<string[]>([]);
  const [updatedIds, setUpdatedIds] = useState<string[]>([]);
  const [focusedId,setFocusedId]=useState<string>();
  const [message, setMessage] = useState("");
  const [sending, setSending] = useState(false);
  const [status, setStatus] = useState("Ready — context and durable tree will both be searched.");
  const [error, setError] = useState("");
  const [windowTokens, setWindowTokens] = useState(100000);
  const [profile, setProfile] = useState("");
  const [verbose, setVerbose] = useState(false);
  const [traceEvents, setTraceEvents] = useState<TraceEvent[]>([]);
  const [traceId, setTraceId] = useState("");
  const [recentTraces, setRecentTraces] = useState<any[]>([]);
  const uploadRef = useRef<HTMLInputElement>(null);

  const refreshTree = async () => setNodes(await api<TreeNode[]>("/personal-tree"));
  const inspect = async (id: string) => {
    try { setSelected(await api(`/personal-tree/nodes/${id}`)); } catch (item) { setError(String(item)); }
  };
  const chooseConversation = async (item: Conversation) => {
    setConversation(item); setTurns([]);
    const context = await api<any>(`/conversations/${item.id}/context`);
    setWindowTokens(context.window_tokens); setProfile(context.conversation.profile_summary);
  };
  const refreshConversations = async () => {
    const items = await api<Conversation[]>("/conversations");
    if (!items.length) {
      const created = await api<Conversation>("/conversations", { method: "POST" });
      setConversations([created]); await chooseConversation(created); return;
    }
    setConversations(items);
    if (!conversation) await chooseConversation(items[0]);
  };
  useEffect(() => {
    setVerbose(localStorage.getItem("treemind-verbose") === "true");
    Promise.all([refreshTree(), refreshConversations()]).catch(item => setError(String(item)));
  }, []);

  const ancestors = (id: string) => { const result: string[]=[]; let item=nodes.find(node=>node.id===id); while(item){result.push(item.id);item=item.parent_id?nodes.find(node=>node.id===item!.parent_id):undefined;} return result; };
  const retrievalPath=useMemo(()=>new Set(routeIds.flatMap(ancestors)),[nodes,routeIds]);
  const storagePath=useMemo(()=>new Set(updatedIds.flatMap(ancestors)),[nodes,updatedIds]);
  const focusedSet=useMemo(()=>focusedId?new Set(ancestors(focusedId)):new Set<string>(),[focusedId,nodes]);
  const flowNodes = useMemo(() => { const groups=new Map<number,TreeNode[]>();nodes.forEach(node=>groups.set(node.depth,[...(groups.get(node.depth)||[]),node]));return nodes.map(node=>{const group=groups.get(node.depth)||[];const index=group.indexOf(node);const root=!node.parent_id;const state=storagePath.has(node.id)?"stored":retrievalPath.has(node.id)?"retrieved":"";return {id:node.id,type:"neural",position:{x:root?560:100+index*(Math.min(230,1160/Math.max(1,group.length))),y:30+node.depth*185},data:{label:node.label,meta:root?`${conversation?.active_tokens?.toLocaleString()||0} / ${windowTokens.toLocaleString()} tokens`:`L${node.depth} · ${node.node_type}`,root,state,focused:focusedId===node.id,dimmed:!!focusedId&&!focusedSet.has(node.id)&&node.parent_id!==focusedId,count:root?"∞":node.depth,onInspect:()=>{setFocusedId(node.id);inspect(node.id)}}};});},[nodes,retrievalPath,storagePath,conversation,windowTokens,focusedId,focusedSet]);
  const flowEdges = useMemo(() => nodes.filter(node=>node.parent_id).map((node,index)=>{const retrieve=retrievalPath.has(node.id)&&retrievalPath.has(node.parent_id!);const store=storagePath.has(node.id)&&storagePath.has(node.parent_id!);return {id:`${node.parent_id}-${node.id}`,source:node.parent_id!,target:node.id,type:"energy",data:{mode:store?"store":retrieve?"retrieve":"idle",delay:index}};}),[nodes,retrievalPath,storagePath]);

  const toggleVerbose = () => setVerbose(value => { const next = !value; localStorage.setItem("treemind-verbose", String(next)); return next; });
  const syncCurrent = async () => { if(!conversation||sending)return;setSending(true);setUpdatedIds([]);setStatus("Synchronizing current chat with TreeMindDB…");try{const response=await fetch(`${API_BASE_URL}/conversations/${conversation.id}/sync/stream`,{method:"POST",headers:{Accept:"text/event-stream"}});const text=await response.text();for(const block of text.split("\n\n")){const type=block.match(/^event: (.+)$/m)?.[1],raw=block.match(/^data: (.+)$/m)?.[1];if(type&&raw)await consumeEvent(type,JSON.parse(raw));}}catch(item){setError(String(item));}finally{setSending(false);}};
  const newConversation = async () => { if(conversation)await syncCurrent(); const item = await api<Conversation>("/conversations", { method: "POST" }); setConversations(items => [item, ...items]); await chooseConversation(item); };
  const resolveUpdate = async (id: string, approved: boolean) => { await api(`/personal-tree/updates/${id}/${approved ? "confirm" : "reject"}`, { method: "POST" }); setStatus(approved ? "Fact replacement approved" : "Current fact retained"); await refreshTree(); if (selected) await inspect(selected.node.id); };
  const openRecent = async () => setRecentTraces(await api<any[]>("/traces?limit=20"));
  const openTrace = async (id: string) => { const data = await api<any>(`/traces/${id}`); setTraceId(id); setTraceEvents(data.events.map((item: any) => ({ event: item.event, payload: item.payload }))); };

  const consumeEvent = async (type: string, data: any) => {
    if (data.trace_id) { setTraceId(data.trace_id); setTraceEvents(items => [...items, { event: type, payload: data }]); }
    if (data.status) setStatus(data.status);
    if (type === "laya_route_selected") {
      setRouteIds(data.selected_node_ids || []);
      setStatus(data.fallback ? "Laya unavailable — visible semantic routing fallback" : "Laya selected tree branches for retrieval");
    }
    if (type === "storage_route_selected" && data.selected_node_id && data.selected_node_id !== "__create_new__") setUpdatedIds(items => [...new Set([...items, data.selected_node_id])]);
    if (type === "fact_stored" && data.node_id) setUpdatedIds(items => [...new Set([...items, data.node_id])]);
    if (type === "answer_ready") setTurns(items => [...items, { role: "assistant", text: data.answer, sources: `${data.used_context ? "Active context" : ""}${data.used_context && data.used_tree ? " + " : ""}${data.used_tree ? "Personal tree" : ""}`, traceId: data.trace_id }]);
    if (type === "branch_created") setStatus(`Created category: ${data.label}`);
    if (type === "fact_update_pending") setTurns(items => items.map((item, index) => index === items.length - 1 && item.role === "assistant" ? { ...item, pending: [...(item.pending || []), data.update_id] } : item));
    if (type === "memory_sync_complete") { setUpdatedIds(data.created_node_ids || []); await refreshTree(); }
    if (type === "profile_summarized") setProfile(data.profile_summary);
    if (type === "context_status") setConversation(item => item ? { ...item, active_tokens: data.active_tokens } : item);
  };

  const send = async (event: FormEvent) => {
    event.preventDefault(); if (!message.trim() || sending || !conversation) return;
    const user = message.trim(); setMessage(""); setSending(true); setError(""); setStatus("Saving message and searching context + tree…"); setTraceEvents([]); setTraceId(""); setRouteIds([]); setUpdatedIds([]); setTurns(items => [...items, { role: "user", text: user }]);
    try {
      const response = await fetch(`${API_BASE_URL}/context-assistant/stream`, { method: "POST", headers: { "Content-Type": "application/json", Accept: "text/event-stream" }, body: JSON.stringify({ message: user, conversation_id: conversation.id }) });
      if (!response.ok || !response.body) throw new Error(await response.text());
      const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "";
      while (true) {
        const { value, done } = await reader.read(); buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
        let at: number;
        while ((at = buffer.indexOf("\n\n")) >= 0) {
          const block = buffer.slice(0, at); buffer = buffer.slice(at + 2);
          const type = block.match(/^event: (.+)$/m)?.[1], raw = block.match(/^data: (.+)$/m)?.[1];
          if (type && raw) await consumeEvent(type, JSON.parse(raw));
        }
        if (done) break;
      }
    } catch (item) { setError(String(item)); } finally { setSending(false); }
  };

  const uploadSource = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]; if (!file) return;
    setSending(true); setError(""); setStatus(`Extracting and classifying ${file.name}…`); setUpdatedIds([]);
    try {
      const form = new FormData(); form.append("file", file);
      const response = await fetch(`${API_BASE_URL}/personal-tree/sources/upload`, { method: "POST", body: form });
      if (!response.ok) throw new Error(await response.text());
      const result = await response.json();
      setUpdatedIds(result.created_nodes?.map((node: TreeNode) => node.id) || []);
      setStatus(`Stored ${result.added_item_ids?.length || 0} fact(s) from ${file.name}`); await refreshTree();
    } catch (item) { setError(String(item)); } finally { setSending(false); if (uploadRef.current) uploadRef.current.value = ""; }
  };

  return <main className="workspace-shell">
    <header className="app-header"><div><h1>TreeMind<span>DB</span></h1><p>Immediate personal memory, with evidence you can inspect.</p></div><div className="header-actions"><button onClick={toggleVerbose} className={verbose ? "toggle active" : "toggle"}>Verbose mode: {verbose ? "On" : "Off"}</button><button onClick={openRecent} className="toggle">Recent traces</button><div className="model-status ready"><b>Active context</b><br/><small>{conversation ? `${conversation.active_tokens.toLocaleString()} / ${windowTokens.toLocaleString()} tokens` : "Creating conversation…"}</small></div></div></header>
    {error && <p className="error">{error}</p>}
    <div className="context-bar"><select value={conversation?.id || ""} onChange={event => { const found = conversations.find(item => item.id === event.target.value); if (found) chooseConversation(found); }}>{conversations.map(item => <option key={item.id} value={item.id}>{item.title} · {item.active_tokens} tokens</option>)}</select><button disabled={sending} onClick={syncCurrent}>Sync TreeMindDB</button><button disabled={sending} onClick={newConversation}>New conversation</button><div><b>Current session summary:</b> {profile || "Chat is held in active context until you sync it."}</div></div>
    {verbose && <section className="trace-panel"><div><h3>Operation trace {traceId && `· ${traceId.slice(0, 8)}`}</h3><p>Actual retrieval paths, Laya choices, fallback reasons, and storage actions. Message text is not copied into traces.</p></div>{traceEvents.map((item, index) => <details key={index} open={index === traceEvents.length - 1}><summary>{item.event}</summary><pre>{JSON.stringify(item.payload, null, 2)}</pre></details>)}{recentTraces.length > 0 && <div className="recent-traces"><h4>Recent traces</h4>{recentTraces.map(trace => <button key={trace.id} onClick={() => openTrace(trace.id)}>{trace.operation_type} · {trace.status} · {trace.id.slice(0, 8)}</button>)}</div>}</section>}
    <div className="split-view">
      <section className="tree-pane"><div className="pane-title"><div><h2>Personal Knowledge Tree</h2><p><i className="legend retrieve"/> Retrieval path <i className="legend store"/> Storage path</p></div><span>{nodes.length} nodes</span></div><div className="tree-canvas"><ReactFlow nodes={flowNodes} edges={flowEdges} fitView><Background/><Controls/></ReactFlow></div><div className="node-detail"><h3>{selected?.node?.label || "Select a branch"}</h3>{selected ? <><p>{selected.node.summary}</p><h4>Current facts</h4>{selected.current_facts?.map((item: any) => <article className="fact" key={item.id}><b>{item.topic}</b><span>{item.content}</span>{item.source_excerpt && <small>Evidence: {item.source_excerpt}</small>}</article>)}<h4>Children</h4><p>{selected.children?.map((child: any) => child.label).join(", ") || "None"}</p>{selected.pending_updates?.length > 0 && <><h4>Pending changes</h4>{selected.pending_updates.map((item: any) => <p key={item.id}>{item.topic}: {item.content}</p>)}</>}{selected.history?.length > 0 && <><h4>Fact history</h4>{selected.history.map((item: any) => <p key={item.id}><s>{item.topic}: {item.content}</s></p>)}</>}</> : <p>Hover for a summary; click for structured facts, source evidence, history, and pending replacements.</p>}</div></section>
      <section className="chat-pane"><div className="pane-title"><div><h2>Personal Assistant</h2><p>Every question checks active context and durable memory in parallel.</p></div><button className="toggle" onClick={() => uploadRef.current?.click()}>Ingest document</button><input ref={uploadRef} className="hidden-input" type="file" accept=".txt,.md,.markdown,.csv,.json,.pdf" onChange={uploadSource}/></div><div className="sync-status"><span className={sending ? "pulse" : ""}/><b>{status}</b></div><div className="messages">{turns.length === 0 && <div className="empty-state"><b>Paste a resume or start talking</b><p>TreeMind saves explicit durable facts after each reply and shows where they land.</p></div>}{turns.map((turn, index) => <article className={`message ${turn.role}`} key={index}><small>{turn.role === "user" ? "You" : "TreeMind"}{turn.sources && ` · ${turn.sources}`}</small><p>{turn.text}</p>{turn.traceId && <button className="trace-link" onClick={() => openTrace(turn.traceId!)}>Inspect retrieval + storage trace</button>}{turn.pending?.map(id => <div className="pending" key={id}><b>Conflicting fact detected</b><button onClick={() => resolveUpdate(id, true)}>Replace</button><button onClick={() => resolveUpdate(id, false)}>Keep current</button></div>)}</article>)}{sending && <article className="message assistant"><small>TreeMind</small><p>Searching evidence and synchronizing your personal tree…</p></article>}</div><form className="chat-input" onSubmit={send}><textarea rows={3} value={message} onChange={event => setMessage(event.target.value)} placeholder="Paste a resume, share an update, or ask anything. Explicit durable facts are classified into the tree."/><button disabled={sending}>{sending ? "Working…" : "Send"}</button></form></section>
    </div>
  </main>;
}
