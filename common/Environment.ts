import type { Position } from "./types.ts";

type EnvironmentSettings = {
  boardSize: number;
}

export default class Environment {
  constructor(private readonly settings: EnvironmentSettings) {}

  get boardSize() {
    return this.settings.boardSize;
  }

  randomPos(): Position {
    return [Math.round((Math.random() * (this.settings.boardSize - 1))), Math.round((Math.random() * (this.settings.boardSize - 1)))];
  }
}
