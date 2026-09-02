import { Router } from "express";
import { nanoid } from "nanoid";
import { z } from "zod";
import { db } from "../db.js";
import { rowToTodo, type TodoRow } from "../types.js";

export const todosRouter = Router();

const createSchema = z.object({
  title: z.string().trim().min(1).max(500),
  notes: z.string().max(5000).optional().default(""),
  priority: z.enum(["low", "medium", "high"]).optional().default("medium"),
  dueDate: z.string().nullable().optional().default(null),
  tags: z.array(z.string().trim().min(1).max(50)).optional().default([]),
});

const updateSchema = z.object({
  title: z.string().trim().min(1).max(500).optional(),
  notes: z.string().max(5000).optional(),
  completed: z.boolean().optional(),
  priority: z.enum(["low", "medium", "high"]).optional(),
  dueDate: z.string().nullable().optional(),
  tags: z.array(z.string().trim().min(1).max(50)).optional(),
});

const reorderSchema = z.object({
  orderedIds: z.array(z.string()).min(1),
});

// GET /api/todos?search=&completed=&priority=&tag=&sort=
todosRouter.get("/", (req, res) => {
  const { search, completed, priority, tag, sort } = req.query as Record<string, string | undefined>;

  let query = "SELECT * FROM todos WHERE 1=1";
  const params: Record<string, string | number | null> = {};

  if (search) {
    query += " AND (title LIKE @search OR notes LIKE @search)";
    params.search = `%${search}%`;
  }
  if (completed === "true" || completed === "false") {
    query += " AND completed = @completed";
    params.completed = completed === "true" ? 1 : 0;
  }
  if (priority && ["low", "medium", "high"].includes(priority)) {
    query += " AND priority = @priority";
    params.priority = priority;
  }
  if (tag) {
    query += " AND tags LIKE @tag";
    params.tag = `%"${tag}"%`;
  }

  const sortMap: Record<string, string> = {
    manual: "position ASC",
    dueDate: "(dueDate IS NULL) ASC, dueDate ASC",
    priority: "CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END ASC",
    created: "createdAt DESC",
    alphabetical: "title COLLATE NOCASE ASC",
  };
  query += ` ORDER BY ${sortMap[sort ?? "manual"] ?? sortMap.manual}`;

  const rows = db.prepare(query).all(params) as unknown as TodoRow[];
  res.json(rows.map(rowToTodo));
});

// GET /api/todos/stats
todosRouter.get("/stats", (_req, res) => {
  const total = (db.prepare("SELECT COUNT(*) c FROM todos").get() as { c: number }).c;
  const completed = (db.prepare("SELECT COUNT(*) c FROM todos WHERE completed = 1").get() as { c: number }).c;
  const overdue = (
    db
      .prepare("SELECT COUNT(*) c FROM todos WHERE completed = 0 AND dueDate IS NOT NULL AND dueDate < date('now')")
      .get() as { c: number }
  ).c;
  res.json({ total, completed, active: total - completed, overdue });
});

// POST /api/todos
todosRouter.post("/", (req, res) => {
  const parsed = createSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.flatten() });
  }
  const { title, notes, priority, dueDate, tags } = parsed.data;
  const now = new Date().toISOString();
  const maxPos = db.prepare("SELECT MAX(position) as m FROM todos").get() as { m: number | null };
  const position = (maxPos.m ?? 0) + 1;
  const id = nanoid(10);

  db.prepare(
    `INSERT INTO todos (id, title, notes, completed, priority, dueDate, tags, position, createdAt, updatedAt)
     VALUES (@id, @title, @notes, 0, @priority, @dueDate, @tags, @position, @createdAt, @updatedAt)`
  ).run({
    id,
    title,
    notes,
    priority,
    dueDate,
    tags: JSON.stringify(tags),
    position,
    createdAt: now,
    updatedAt: now,
  });

  const row = db.prepare("SELECT * FROM todos WHERE id = ?").get(id) as unknown as TodoRow;
  res.status(201).json(rowToTodo(row));
});

// POST /api/todos/reorder
todosRouter.post("/reorder", (req, res) => {
  const parsed = reorderSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.flatten() });
  }
  const { orderedIds } = parsed.data;
  const update = db.prepare("UPDATE todos SET position = @position, updatedAt = @updatedAt WHERE id = @id");
  const now = new Date().toISOString();
  db.exec("BEGIN");
  try {
    orderedIds.forEach((id, index) => update.run({ id, position: index + 1, updatedAt: now }));
    db.exec("COMMIT");
  } catch (err) {
    db.exec("ROLLBACK");
    throw err;
  }
  res.json({ ok: true });
});

// DELETE /api/todos/completed
todosRouter.delete("/completed", (_req, res) => {
  const result = db.prepare("DELETE FROM todos WHERE completed = 1").run();
  res.json({ deleted: result.changes });
});

// PATCH /api/todos/:id
todosRouter.patch("/:id", (req, res) => {
  const parsed = updateSchema.safeParse(req.body);
  if (!parsed.success) {
    return res.status(400).json({ error: parsed.error.flatten() });
  }
  const existing = db.prepare("SELECT * FROM todos WHERE id = ?").get(req.params.id) as unknown as TodoRow | undefined;
  if (!existing) return res.status(404).json({ error: "Todo not found" });

  const merged = {
    id: existing.id,
    title: parsed.data.title ?? existing.title,
    notes: parsed.data.notes ?? existing.notes,
    completed: parsed.data.completed !== undefined ? (parsed.data.completed ? 1 : 0) : existing.completed,
    priority: parsed.data.priority ?? existing.priority,
    dueDate: parsed.data.dueDate !== undefined ? parsed.data.dueDate : existing.dueDate,
    tags: parsed.data.tags !== undefined ? JSON.stringify(parsed.data.tags) : existing.tags,
    updatedAt: new Date().toISOString(),
  };

  db.prepare(
    `UPDATE todos SET title=@title, notes=@notes, completed=@completed, priority=@priority,
     dueDate=@dueDate, tags=@tags, updatedAt=@updatedAt WHERE id=@id`
  ).run(merged);

  const row = db.prepare("SELECT * FROM todos WHERE id = ?").get(req.params.id) as unknown as TodoRow;
  res.json(rowToTodo(row));
});

// DELETE /api/todos/:id
todosRouter.delete("/:id", (req, res) => {
  const result = db.prepare("DELETE FROM todos WHERE id = ?").run(req.params.id);
  if (result.changes === 0) return res.status(404).json({ error: "Todo not found" });
  res.status(200).json({ ok: true });
});
