from gym.envs.registration import register

register(
    id='bank_world-v0',
    entry_point='gym_bank_world.envs:BankWorldEnv',
)