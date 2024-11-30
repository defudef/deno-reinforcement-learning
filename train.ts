import { BOARD_SIZE } from "./common/const.ts";
import Model from "./common/Model.ts";
import Board from "./common/Board.ts";
import Agent from "./common/Agent.ts";
import Goal from "./common/Goal.ts";
import Environment from "./common/Environment.ts";

const train = async (
  model: Model,
  opts: {
    startingEpoch: number;
    epochs: number;
    boardSize: number;
    maxSteps: number;
    startingEpsilon?: number;
  },
  env: Environment,
): Promise<{ currentEpsilon: number; endingEpoch: number; wins: number }> => {
  const { epochs, startingEpoch, maxSteps, startingEpsilon = 0.3 } = opts;

  let epsilon = startingEpsilon;
  let wins = 0;

  for (let epoch = 0; epoch < epochs; epoch++) {
    console.log(epoch + startingEpoch + 1, 'epoch');

    const agent = new Agent(env);
    const goal = new Goal(env.randomPos());
    const board = new Board(env);

    board.addObject(agent);
    board.addObject(goal);

    let reward = 0;

    for (let step = 0; step < maxSteps; step++) {
      // 1. Make agent move
      const currentFeatures = model.toTensor(agent, goal);
      let [action, qValues] = model.predictWithQValues(agent, goal);

      // 2. Epsilon greedy: Randomize if agent should exploit (predicted move) or explore (randomized move)
      action = model.performEpsilonGreedy(action, step);
      agent.move(action);

      // 3. Calculate reward or penalty
      reward = model.calculateReward(agent, goal);

      // 4. If agent position hasn't changed, randomize action again (exploration will help agent to learn that they can't go through walls)
      if (agent.isIdle()) {
        action = model.getRandomAction();
        agent.undoMove();
        agent.move(action);
      }

      // 5. Predict next Q values (next movement)
      const [nextAction, nextQValues] = model.predictWithQValues(agent, goal);
      const maxNextQValue = model.actionToQValue(nextAction, nextQValues);

      const targetQ = model.calculateTargetQValues(
        qValues,
        action,
        reward,
        maxNextQValue,
      )

      // 6. Train model
      await model.fit(currentFeatures, targetQ);

      if (agent.hasReachedGoal(goal)) {
        wins++;
        console.log('Goal reached!');
        break;
      }
    }



    Deno.stdout.writeSync(new TextEncoder().encode(`(r: ${reward})`));
  }

  return { currentEpsilon: epsilon, endingEpoch: epochs + startingEpoch, wins };
};

const evaluate = async (model: Model, maxMoves: number, env: Environment, wins: [number, number], epochNo: number, successfulEvaluations: number): Promise<boolean | void> => {
  const delay = 50; // ms between each step

  const agent = new Agent(env);
  const goal = new Goal(env.randomPos());
  const board = new Board(env);

  board.addObject(agent);
  board.addObject(goal);

  board.render();

  for (let i = 0; i < maxMoves; i++) {
    const [action, res] = model.predictWithQValues(agent, goal);

    agent.move(action);

    board.render();

    if (agent.hasReachedGoal(goal)) {
      return true;
    }

    console.log('Wins:', successfulEvaluations);
    console.log('Epochs:', epochNo);
    console.log('Moves:', i + 1);
    console.log('Error rate:', 1 - wins[0] / wins[1], '(the lower the better)');
    console.log(res);

    // wait 100ms
    await new Promise(resolve => setTimeout(resolve, delay));
  }

  return false;
};

async function main() {
  await Model.init();

  const model = new Model();
  const env = new Environment({
    boardSize: BOARD_SIZE,
  });

  const maxMoves = 50;
  const epochs = 25;

  let epsilon = 0.2;
  let epochNo = 0;

  let prevWins = 0;

  while (true) {
    const weights = model.getWeights();
    
    const res = await train(model, {
      epochs,
      startingEpoch: epochNo,
      boardSize: BOARD_SIZE,
      maxSteps: maxMoves,
      startingEpsilon: epsilon
    }, env);

    if (res.wins >= prevWins) {
      prevWins = res.wins;
    } else {
      model.setWeights(weights); // Rollback model weights if it didn't improve
      model.increaseExplorationRate(0.2); // Increase exploration rate to help agent to find more possibilities
    }

    epsilon = res.currentEpsilon;
    epochNo = res.endingEpoch;

    let wins: number = 0;

    do {
      const hasWon = await evaluate(model, maxMoves, env, [prevWins, epochs], epochNo, wins) ?? false;

      wins = hasWon ? wins + 1 : 0;
    } while (wins > 0);
  }
}

main();