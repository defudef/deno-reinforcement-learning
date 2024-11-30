import '@tensorflow/tfjs-backend-wasm';
import * as tf from '@tensorflow/tfjs';
import type Agent from "./Agent.ts";
import type Goal from "./Goal.ts";
import type { Action } from "./types.ts";
import { actions } from "./const.ts";

type ModelHyperParams = {
  learningRate?: number;
  rewardDiscount?: number;
  actionEpsilon?: number;
  maxActionEpsilon?: number;
  rewardStrategy?: {
    movement: number; // penalty for each move
    gettingFurther: number; // additional penalty for moving away from the goal
    goal: number; // reward for reaching the goal
    samePosition: number; // additional penalty for staying in the same position
    normalizingFactor: number; // optional normalizing factor
  }
};

type CreateModelParams = {
  hiddenLayers?: number;
  units?: number;
  batchSize?: number;
}

export const DEFAULT_REWARD_STRATEGY: Required<ModelHyperParams>['rewardStrategy'] = {
  movement: -1,
  gettingFurther: -6,
  goal: 100,
  samePosition: -4,
  normalizingFactor: 1,
};


/**
 * @description Before you train or evaluate model make sure you call and await init() method first!
 * @description Before saving the model, make sure you have server running
 */
export default class Model {
  private model: tf.Sequential;
  private readonly params: Required<ModelHyperParams>;
  
  private batchSize: number = 16;
  private batchFeatures: tf.Tensor[] = [];
  private batchTargets: tf.Tensor[] = [];

  constructor(params?: ModelHyperParams) {
    this.params = {
      learningRate: params?.learningRate ?? 0.001,
      rewardDiscount: params?.rewardDiscount ?? 0.9,
      actionEpsilon: params?.actionEpsilon ?? 0.5,
      rewardStrategy: params?.rewardStrategy ?? { ...DEFAULT_REWARD_STRATEGY },
      maxActionEpsilon: params?.maxActionEpsilon ?? 0.5,
    };
    this.model = this.create();
  }

  static async init() { 
    await tf.ready();
    await tf.setBackend('wasm');
  }

  create(params?: CreateModelParams): tf.Sequential {
    if (params?.batchSize) {
      this.batchSize = params.batchSize;
    }

    this.model = tf.sequential();

    const hiddenLayers = params?.hiddenLayers ?? 5;
    const units = params?.units ?? 128;
    const leakyReLUAlpha = 0.01;
    
    this.model.add(tf.layers.dense({
      inputShape: [4], // Input is: [agent.x, agent.y, goal.x, goal.y]
      units,
    }));
    this.model.add(tf.layers.leakyReLU({ alpha: leakyReLUAlpha })); // everyone loves leaky ReLU

    // Let's add hidden layers to improve model performance
    for (let i = 0; i < hiddenLayers; i++) {
      this.model.add(tf.layers.dense({
        units,
      }));
      this.model.add(tf.layers.leakyReLU({ alpha: leakyReLUAlpha }));
      this.model.add(tf.layers.dropout({ rate: 0.2 }));
    }

    this.model.add(tf.layers.dense({
      activation: 'softmax', // softmax is good fit to get one, most important action of all
      units: 4, // One output for each action (up, down, left, right)
    }));
  
    this.model.compile({
      // optimizer: tf.train.sgd(this.params.learningRate),
      optimizer: tf.train.adam(this.params.learningRate, undefined, undefined, 1e-5),
      loss: tf.metrics.categoricalCrossentropy, // Good loss function for softmax
    });
  
    return this.model;
  }

  async fit(currentFeatures: tf.Tensor, targetFeatures: tf.Tensor): Promise<void> {
    this.batchFeatures.push(currentFeatures);
    this.batchTargets.push(targetFeatures);

    if (this.batchFeatures.length < this.batchSize) {
      return;
    }

    const features = tf.concat(this.batchFeatures);
    const targets = tf.concat(this.batchTargets);

    // Train the model
    await this.model.fit(features, targets);

    this.batchFeatures = [];
    this.batchTargets = [];

    features.dispose();
    targets.dispose();
  }

  predict(agent: Agent, goal: Goal): Action {
    return this.predictWithQValues(agent, goal)[0];
  }

  predictWithQValues(agent: Agent, goal: Goal): [Action, Float32Array] {
    const qValues = this.model
      .predict(tf.tensor([[agent.x, agent.y, goal.x, goal.y]]))
      // deno-lint-ignore ban-ts-comment
      // @ts-ignore
      .dataSync() as Float32Array;

    return [
      actions[qValues.indexOf(Math.max(...qValues))],
      qValues,
    ];
  }

  toTensor(agent: Agent, goal: Goal): tf.Tensor {
    return tf.tensor([[agent.x, agent.y, goal.x, goal.y]]);
  }

  actionToQValue(action: Action, qValues: Float32Array): number {
    return qValues[actions.indexOf(action)];
  }

  performEpsilonGreedy(action: Action, step: number): Action {
    if (Math.random() < this.params.actionEpsilon) {
      return this.getRandomAction();
    }

    // if (step % 10 === 0) {
      this.params.actionEpsilon *= 0.995;
    // }

    return action;
  }

  getRandomAction(): Action {
    return actions[Math.floor(Math.random() * actions.length)];
  }

  setWeights(weights: tf.Tensor[]): void {
    this.model.setWeights(weights);
  }

  getWeights(): tf.Tensor[] {
    return this.model.getWeights();
  }

  /**
   * Usually, this is the function you want to spend most of your time on.
   * Observe the agent behavior and adjust the reward strategy accordingly.
   * Your strategy should help the agent to reach the goal as efficiently as possible.
   * 
   * In this case we are using below strategy:
   * GREAT reward - when agent reaches the goal
   * SMALL penalty - for each move
   * SMALL penalty - if it gets further from the goal
   * A BIT HIGHER penalty - if it stays in the same position (for example blocks on the wall).
   * 
   * Follow these rules when you train with reinforcement learning:
   * - significantly reward for achieving the goal
   * - give it a small penalty for each step (so it needs to figure out how to do it a bit more efficiently)
   * - give it penalty for each other thing that is not helping it to reach the goal (like staying in the same position, moving further from the goal, etc.)
   */
  calculateReward(agent: Agent, goal: Goal): number {
    let reward = 0;

    // We just return this reward because agent has won
    //
    // ♪♪ and nothing else matters... ♪♪
    if (agent.hasReachedGoal(goal)) {
      return this.params.rewardStrategy.goal;
    }

    // Add penalty for movement
    reward += this.params.rewardStrategy.movement;

    // Did agent stay in the same position? If so - give it a penalty
    if (agent.isIdle()) {
      reward += this.params.rewardStrategy.samePosition;
    }

    // Did agent moved further from the goal? If so - give it a penalty
    const pastPosition = agent.getPastPosition(1);

    if (pastPosition && agent.distance(goal.position) > goal.distance(pastPosition)) {
      reward += this.params.rewardStrategy.gettingFurther;
    }

    return reward * this.params.rewardStrategy.normalizingFactor;
  }

  performDiscountedReward(reward: number, targetQ: number[], maxNextQValue: number, action: Action): number {
    // const alpha = 0.1;
    // targetQ[actions.indexOf(action)] =
    //   (1 - alpha) * targetQ[actions.indexOf(action)] +
    //   alpha * (reward + 0.9 * maxNextQValue);

    const alpha = 0.1;

    const maxTargetQ = this.actionToQValue(action, targetQ as unknown as Float32Array);

    return (1 - alpha) * maxTargetQ + alpha * (reward + this.params.rewardDiscount * maxNextQValue);
  }

  calculateTargetQValues(qValues: Float32Array, action: Action, reward: number, maxNextQValue: number): tf.Tensor {
    const targetQ = [...qValues];

    targetQ[actions.indexOf(action)] = this.performDiscountedReward(reward, targetQ, maxNextQValue, action);

    return tf.tensor([targetQ]);
  }

  increaseExplorationRate(rate: number): void {
    this.params.actionEpsilon *= (1 + rate);

    if (this.params.actionEpsilon > this.params.maxActionEpsilon) {
      this.params.actionEpsilon = 1;
    }
  }

  makeCopy(): Model {
    const model = new Model(JSON.parse(JSON.stringify(this.params)));
    
    const weights = model.model.getWeights();

    model.model.setWeights(weights);

    return model;
  }
}
