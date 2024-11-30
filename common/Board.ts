import type BoardObject from "./BoardObject.ts";
import type Environment from "./Environment.ts";

/*
 * TODO:
 * - Multiple agents
 * - Goal as an independent entity able to move
 */

export const BOARD_DEFAULT_SIZE = 10;

export default class Board {
  private readonly matrix: string[][];  

  private readonly objects: BoardObject[] = [];

  constructor(env: Environment) {
    this.matrix = Array.from(
      { length: env.boardSize }, 
      () => Array.from({ length: env.boardSize }, () => '| |')
    );
  }

  addObject(object: BoardObject): void {
    this.objects.push(object);
  }

  render() {
    console.clear();

    // clear board
    this.matrix.forEach((row, i) => {
      row.forEach((_, j) => {
        this.matrix[i][j] = '| |';
      });
    });

    // fill objects
    this.objects.forEach(object => {
      this.matrix[object.x][object.y] = `|${object.symbol}|`;
    });

    // render board
    console.log(this.matrix.map(row => row.join('')).join('\n'));
  }
}
