export type Priority = "low" | "medium" | "high";

export interface Todo {
  id: string;
  title: string;
  notes: string;
  completed: boolean;
  priority: Priority;
  dueDate: string | null;
  tags: string[];
  position: number;
  createdAt: string;
  updatedAt: string;
}

export interface Stats {
  total: number;
  completed: number;
  active: number;
  overdue: number;
}

export type FilterStatus = "all" | "active" | "completed";
export type SortMode = "manual" | "dueDate" | "priority" | "created" | "alphabetical";

export interface TodoInput {
  title: string;
  notes?: string;
  priority?: Priority;
  dueDate?: string | null;
  tags?: string[];
}

export interface TodoUpdate {
  title?: string;
  notes?: string;
  completed?: boolean;
  priority?: Priority;
  dueDate?: string | null;
  tags?: string[];
}
