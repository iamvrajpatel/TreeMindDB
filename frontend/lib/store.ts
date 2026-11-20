"use client";
import {create} from "zustand";
export const useUi=create<{collectionId:string;setCollection:(id:string)=>void;selectedNodeId?:string;selectNode:(id?:string)=>void}>((set)=>({collectionId:"",setCollection:(collectionId)=>set({collectionId}),selectedNodeId:undefined,selectNode:(selectedNodeId)=>set({selectedNodeId})}));
