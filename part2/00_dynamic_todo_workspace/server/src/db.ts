import { DatabaseSync } from "node:sqlite";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const dbPath = path.join(__dirname, "..", "data.sqlite3");

export const db = new DatabaseSync(dbPath);
db.exec("PRAGMA journal_mode = WAL;");

db.exec(`
  CREATE TABLE IF NOT EXISTS todos (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    notes TEXT DEFAULT '',
    completed INTEGER NOT NULL DEFAULT 0,
    priority TEXT NOT NULL DEFAULT 'medium',
    dueDate TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    position REAL NOT NULL,
    createdAt TEXT NOT NULL,
    updatedAt TEXT NOT NULL
  );
`);

const seedCount = db.prepare("SELECT COUNT(*) as c FROM todos").get() as { c: number };

if (seedCount.c === 0) {
  const now = new Date().toISOString();
  const insert = db.prepare(`
    INSERT INTO todos (id, title, notes, completed, priority, dueDate, tags, position, createdAt, updatedAt)
    VALUES (@id, @title, @notes, @completed, @priority, @dueDate, @tags, @position, @createdAt, @updatedAt)
  `);
  const seed = [
    {
      id: "seed-1",
      title: "Welcome to Flow — your new todo list",
      notes: "Click a task to expand it. Try dragging to reorder!",
      completed: 0,
      priority: "medium",
      dueDate: null,
      tags: JSON.stringify(["welcome"]),
      position: 1,
      createdAt: now,
      updatedAt: now,
    },
    {
      id: "seed-2",
      title: "Set a due date and priority on a task",
      notes: "",
      completed: 0,
      priority: "high",
      dueDate: new Date(Date.now() + 86400000).toISOString().slice(0, 10),
      tags: JSON.stringify(["tutorial"]),
      position: 2,
      createdAt: now,
      updatedAt: now,
    },
    {
      id: "seed-3",
      title: "Mark this one complete",
      notes: "",
      completed: 1,
      priority: "low",
      dueDate: null,
      tags: JSON.stringify([]),
      position: 3,
      createdAt: now,
      updatedAt: now,
    },
  ];
  for (const row of seed) insert.run(row);
}
