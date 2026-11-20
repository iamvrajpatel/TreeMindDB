export const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000/api/v1";
export type Collection={id:string;name:string;embedding_dimension:number};
export type Node={id:string;parent_id:string|null;child_ids:string[];node_type:string;depth:number;vector_count:number;document_count:number;label?:string;summary:string;radius:number};
export async function api<T>(path:string,init?:RequestInit):Promise<T>{const r=await fetch(`${API_BASE_URL}${path}`,{headers:{"Content-Type":"application/json",...(init?.headers||{})},...init});if(!r.ok)throw new Error(await r.text());return r.json()}
