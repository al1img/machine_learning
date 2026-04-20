# Reinforcement Learning with Neural Networks

## DQN

### Experiments with dqnagent.py

* adding `torch.nn.utils.clip_grad_value_(self._policy_net.parameters(), 100)` has now effect;
* LR = 1e-3 AdamW optimizer with amsgrad=True seems to be more stable than Adam. Looks like AdamW with amsgrad=True is
  more stable than Adam if LR is high (1e-3), but at lower LR (1e-4) both optimizers perform similarly;
* amsgrad=True seems to be more stable than amsgrad=False;

https://medium.com/@rizvaanpatel/cartpole-in-openai-gym-the-unexpected-rl-breakthrough-you-didnt-see-coming-65adf46ec633
https://medium.com/data-science/an-intuitive-explanation-of-policy-gradient-part-1-reinforce-aa4392cbfd3c
https://medium.com/@rizvaanpatel/step-by-step-guide-implementing-policy-gradient-in-python-for-reinforcement-learning-21b94648f746
https://christianbernecker.medium.com/hands-on-policy-gradients-solving-cartpole-with-reinforce-30a32c2ea408
https://github.com/Finspire13/pytorch-policy-gradient-example
https://github.com/tims457/RL_Agent_Notebooks
https://becominghuman.ai/lets-build-an-atari-ai-part-0-intro-to-rl-9b2c5336e0ec
https://aigreeks.com/solve-cartpole-v1-in-open-gym-reinforcement-learning/
https://medium.com/nerd-for-tech/policy-gradients-reinforce-with-baseline-6c871a3a068
https://medium.com/@sachith.icc/mastering-cartpole-a-complete-journey-through-6-reinforcement-learning-algorithms-06d92fa6d601
https://lilianweng.github.io/posts/2018-04-08-policy-gradient/
https://medium.com/data-science/understanding-actor-critic-methods-931b97b6df3f
https://www.tensorflow.org/tutorials/reinforcement_learning/actor_critic
https://docs.pytorch.org/tutorials/intermediate/reinforcement_ppo.html
https://medium.com/@felix.verstraete/mastering-proximal-policy-optimization-ppo-in-reinforcement-learning-230bbdb7e5e7
https://www.datacamp.com/tutorial/proximal-policy-optimization
https://docs.cleanrl.dev/rl-algorithms/ppo/
https://stable-baselines3.readthedocs.io/en/master/
https://github.com/Chris-hughes10/simple-ppo
https://shivang-ahd.medium.com/generalized-advantage-estimation-a-deep-dive-into-bias-variance-and-policy-gradients-a5e0b3454dad
https://apxml.com/courses/advanced-reinforcement-learning
https://arxiv.org/pdf/1707.06347
https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
