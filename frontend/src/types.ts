export interface GraphNode {
  id: string;
  label: string;
  type?: string;
  entityType?: string | null;
  degree: number;
}

export interface GraphLink {
  id: string;
  source: string;
  target: string;
  predicate: string;
  weight: number;
}

export interface GraphPayload {
  nodes: GraphNode[];
  links: GraphLink[];
}

export interface Passage {
  id: string;
  docTitle: string;
  section: string;
  text: string;
}

export interface RetrievalInfo {
  chunkIds: string[];
  nodeIds: string[];
  edgeIds: string[];
  passages: Passage[];
}

export interface QueryResponse {
  answer: string;
  retrieval: RetrievalInfo;
  subgraph: GraphPayload;
}
