import BoardObject from "./BoardObject.ts";
import type { Position } from "./types.ts";

export default class Goal extends BoardObject {
  constructor(private readonly fixedPosition: Position) {
    super();
  }

  override get position(): Position {
    return this.fixedPosition;
  }

  override get symbol(): string {
    return 'E';
  }
}
