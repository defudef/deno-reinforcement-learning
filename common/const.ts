import type { Action, Position } from "./types.ts";

export const BOARD_SIZE = 10;
export const actions: [Action, Action, Action, Action] = ['up', 'down', 'left', 'right'];
export const actionMap: Record<Action, Position> = {
  up: [-1, 0],
  down: [1, 0],
  left: [0, -1],
  right: [0, 1],
};
