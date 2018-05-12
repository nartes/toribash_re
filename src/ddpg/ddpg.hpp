#pragma once

class DDPG
{
public:
    DDPG();

private:
    const int MAX_EPISODES;
    const int MAX_EP_STEPS;
    const int LR_A;
    const int LR_C;
    const int GAMMA;
    const int TAU;
    const int MEMORY_CAPACITY;
    const int BATCH_SIZE;
};
