class Config:
    # 模型参数
    D_MODEL = 256
    NHEAD = 8
    NUM_LAYERS = 6
    HIDDEN_DIM = 512
    
    # 强化学习参数
    DISCOUNT_FACTOR = 0.99
    LEARNING_RATE = 0.001
    BATCH_SIZE = 64
    DROPOUT_RATE = 0.5
    
    # 训练参数
    TRAIN_RATIO = 0.7
    LAMBDA2 = 0.7
    MAX_STEPS = 20
    
    # 经验回放
    REPLAY_BUFFER_SIZE = 10000
    TARGET_UPDATE_FREQ = 100
    
    # 路径设置
    EMBEDDING_DIM = 256