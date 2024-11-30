import BoardObject from "./BoardObject.ts";
import type Environment from "./Environment.ts";
import type Goal from "./Goal.ts";
import type { Action } from "./types.ts";
import type { Position } from "./types.ts";

const X = 0;
const Y = 1;

export default class Agent extends BoardObject {
  private readonly moveHistory: Position[] = [];

  constructor(
    private readonly env: Environment,
    startingPosition?: Position,
  ) {
    super();

    this.moveHistory.push(startingPosition ?? env.randomPos());
  }

  override get position(): Position {
    return this.moveHistory[this.moveHistory.length - 1];
  }

  override get symbol(): string {
    return 'o';
  }

  getPastPosition(stepsBack: number): Position | undefined {
    const index = this.moveHistory.length - 1 - stepsBack;

    if (index < 0) {
      return undefined;
    }

    return this.moveHistory[this.moveHistory.length - 1 - stepsBack];
  }

  undoMove(): void {
    this.moveHistory.pop();
  }

  move(direction: Action): void {
    const position = this.position;
    const newPosition: Position = [...position];

    switch (direction) {
      case 'up':
        newPosition[Y]--;
        break;
      case 'down':
        newPosition[Y]++;
        break;
      case 'left':
        newPosition[X]--;
        break;
      case 'right':
        newPosition[X]++;
        break
    }

    this.moveHistory.push(
      this.validatePosition(newPosition)
        ? newPosition
        : position
    );
  }

  hasReachedGoal(goal: Goal): boolean {
    return this.position.toString() === goal.position.toString();
  }

  /**
   * @returns true if the agent current and last position remain the same
   */
  isIdle(): boolean {
    const pastPos = this.getPastPosition(1);

    return pastPos 
      ? this.position.toString() === pastPos.toString()
      : false;
  }

  private validatePosition(newPosition: Position): boolean {
    switch (true) {
      case newPosition[X] < 0:
      case newPosition[X] >= this.env.boardSize:
      case newPosition[Y] < 0:
      case newPosition[Y] >= this.env.boardSize:
        return false;
    }

    return true;
  }
}
