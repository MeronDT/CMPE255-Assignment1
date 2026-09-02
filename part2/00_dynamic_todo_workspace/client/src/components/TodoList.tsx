import {
  DndContext,
  type DragEndEvent,
  KeyboardSensor,
  PointerSensor,
  closestCenter,
  useSensor,
  useSensors,
} from "@dnd-kit/core";
import { SortableContext, sortableKeyboardCoordinates, verticalListSortingStrategy } from "@dnd-kit/sortable";
import type { Todo, TodoUpdate } from "../types";
import { TodoItem } from "./TodoItem";
import { EmptyState } from "./EmptyState";

interface Props {
  todos: Todo[];
  onUpdate: (id: string, update: TodoUpdate) => void;
  onDelete: (id: string) => void;
  onReorder: (orderedIds: string[]) => void;
  dragDisabled: boolean;
  emptyKind: "none" | "no-results" | "all-done";
}

export function TodoList({ todos, onUpdate, onDelete, onReorder, dragDisabled, emptyKind }: Props) {
  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 6 } }),
    useSensor(KeyboardSensor, { coordinateGetter: sortableKeyboardCoordinates })
  );

  if (todos.length === 0) {
    return <EmptyState kind={emptyKind} />;
  }

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    const oldIndex = todos.findIndex((t) => t.id === active.id);
    const newIndex = todos.findIndex((t) => t.id === over.id);
    const reordered = [...todos];
    const [moved] = reordered.splice(oldIndex, 1);
    reordered.splice(newIndex, 0, moved);
    onReorder(reordered.map((t) => t.id));
  };

  return (
    <DndContext sensors={sensors} collisionDetection={closestCenter} onDragEnd={handleDragEnd}>
      <SortableContext items={todos.map((t) => t.id)} strategy={verticalListSortingStrategy}>
        <ul className="space-y-2">
          {todos.map((todo) => (
            <TodoItem key={todo.id} todo={todo} onUpdate={onUpdate} onDelete={onDelete} dragDisabled={dragDisabled} />
          ))}
        </ul>
      </SortableContext>
    </DndContext>
  );
}
