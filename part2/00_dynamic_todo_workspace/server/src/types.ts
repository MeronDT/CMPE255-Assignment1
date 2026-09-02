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

export interface TodoRow {
  id: string;
  title: string;
  notes: string;
  completed: number;
  priority: Priority;
  dueDate: string | null;
  tags: string;
  position: number;
  createdAt: string;
  updatedAt: string;
}

export function rowToTodo(row: TodoRow): Todo {
  return {
    ...row,
    completed: !!row.completed,
    tags: JSON.parse(row.tags || "[]"),
  };
}
