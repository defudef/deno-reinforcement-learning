import type { Position } from "./types.ts";

const X = 0;
const Y = 1;

export default abstract class BoardObject {
  get x(): number {
    return this.position[X];
  }

  get y(): number {
    return this.position[Y];
  }

  abstract get position(): Position;

  abstract get symbol(): string;

  distance(position: Position): number {
    const [x1, y1] = this.position;
    const [x2, y2] = position;

    return Math.abs(y1 - y2) + Math.abs(x1 - x2);
  }
}
