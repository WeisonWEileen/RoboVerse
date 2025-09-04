from humanoid_visualrl.actor_critic.actor_critic_cnn import ActorCriticCNN

class ActorCriticCNNGRU(ActorCriticCNN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.memory_a = Memory(self.num_actor_obs, type="gru", num_layers=self.rnn_num_layers, hidden_size=self.rnn_hidden_dim)
        self.memory_c = Memory(self.num_critic_obs, type="gru", num_layers=self.rnn_num_layers, hidden_size=self.rnn_hidden_dim)