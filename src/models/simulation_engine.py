import numpy as np

from models.engine import BaseEngine
from utils.history_utils import TimeSeries, TransitionHistory
import utils.global_configs as global_configs


EXPECTED_NUM_DAYS = 300 

class SimulationEngine(BaseEngine):

    states = []
    num_states = len(states)
    state_str_dict = {}

    ext_code = 0

    transitions = []
    num_transitions =  len(transitions)

    final_states = []     # no transition from them 
    invisible_states = [] # does not count into population 
    unstable_states = []  # can change 


    fixed_model_parameters = {}
    model_parameters = {} 

    common_arguments = {
        "random_seed": (None, "random seed value"),
        "start_day": (1, "day to start")
    }
    

    def __init__(self, G, **kwargs):

        self.G = G  # backward compatibility
        self.graph = G

        self.init_kwargs = kwargs

        # 2. model initialization
        self.inicialization()

        # 3. time and history setup
        self.setup_series_and_time_keeping()

        # 4. init states and their counts
        self.states_and_counts_init(ext_nodes=self.num_ext_nodes,
                                    ext_code=self.ext_code)


        # 5. set callback to None
        self.periodic_update_callback = None

        self.T = self.start_day - 1

    def update_graph(self, new_G):
        if new_G is not None:
            self.G = new_G  # just for backward compability
            self.graph = new_G
            self.num_nodes = self.graph.num_nodes
            try:
                self.num_ext_nodes = self.graph.num_nodes - self.graph.num_base_nodes
            except AttributeError:
                #  for saved old graph
                self.num_ext_nodes = 0
            self.nodes = np.arange(self.graph.number_of_nodes).reshape(-1, 1)



    def inicialization(self):

        super().inicialization()

        # node indexes
        self.nodes = np.arange(self.graph.num_nodes).reshape(-1, 1)
        self.num_nodes = self.graph.num_nodes

        

    def setup_series_and_time_keeping(self):

        super().setup_series_and_time_keeping()
        

        tseries_len = self.num_transitions * self.num_nodes

        self.tseries = TimeSeries(tseries_len, dtype=float)
        self.history = TransitionHistory(tseries_len)

        # state history
        if global_configs.SAVE_NODES:
            history_len = EXPECTED_NUM_DAYS
        else:
            history_len = 1
        self.states_history = TransitionHistory(
            history_len, width=self.num_nodes)

        if global_configs.SAVE_DURATIONS:
            self.states_durations = {
                s: []
                for s in self.states
            }

        self.durations = np.zeros(self.num_nodes, dtype=int)

        # state_counts ... numbers of inidividuals in given states
        self.state_counts = {
            state: TimeSeries(EXPECTED_NUM_DAYS, dtype=int)
            for state in self.states
        }

        self.state_increments = {
            state: TimeSeries(EXPECTED_NUM_DAYS, dtype=int)
            for state in self.states
        }

        # N ... actual number of individuals in population
        self.N = TimeSeries(EXPECTED_NUM_DAYS, dtype=float)


    def states_and_counts_init(self, ext_nodes=None, ext_code=None):
        super().states_and_counts_init(ext_nodes, ext_code)

        # time to go until I move to the state state_to_go
        self.time_to_go = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)
        self.state_to_go = np.full(
            self.num_nodes, fill_value=-1, dtype="int32").reshape(-1, 1)

        self.current_state = self.states_history[0].copy().reshape(-1, 1)


        # need update = need to recalculate time to go and state_to_go
        self.need_update = np.ones(self.num_nodes, dtype=bool)

        
    def daily_update(self, nodes):
        """
        Everyday checkup
        """
        pass 


    def change_states(self, nodes, target_state=None):
        """
        nodes that just entered a new state, update plan
        """
        # discard current state
        self.memberships[:, nodes == True] = 0


        for node in nodes.nonzero()[0]:
            if target_state is None:
                new_state = self.state_to_go[node][0]
            else:
                new_state = target_state
            old_state = self.current_state[node, 0]

            self.memberships[new_state, node] = 1
            self.state_counts[new_state][self.t] += 1
            self.state_counts[old_state][self.t] -= 1
            self.state_increments[new_state][self.t] += 1
            if global_configs.SAVE_NODES:
                self.states_history[self.t][node] = new_state

        if target_state is None:
            self.current_state[nodes] = self.state_to_go[nodes]
        else:
            self.current_state[nodes] = target_state
        self.update_plan(nodes)

    def update_plan(self, nodes):
        """ This is done for nodes that  just changed their states.
        New plans are generated according the state."""
        pass

    def _get_target_nodes(self, nodes, state):
        ret = nodes.copy().ravel()
        is_target_state = self.memberships[state, ret, 0]
        ret[nodes.flatten()] = is_target_state
        # ret = np.logical_and(
        #     self.memberships[state].flatten(),
        #     nodes.flatten()
        # )
        return ret
    
    def print(self, verbose=False):
        print(f"T = {self.T} ({self.t})")
        if verbose:
            for state in self.states:
                print(f"\t {self.state_str_dict[state]} = {self.state_counts[state][self.t]}")

    def save_durations(self, f):
        for s in self.states:
            line = ",".join([str(x) for x in self.states_durations[s]])
            print(f"{self.state_str_dict[s]},{line}", file=f)

    def save_node_states(self, filename):
        if global_configs.SAVE_NODES is False:
            logging.warning(
                "Nodes states were not saved, returning empty data frame.")
            return pd.DataFrame()
        index = range(0, self.t+1)
        columns = self.states_history.values
        df = pd.DataFrame(columns, index=index)
        df.to_csv(filename)
        # df = df.replace(self.state_str_dict)
        # df.to_csv(filename)
        # print(df)

    def to_df(self):

        df = super().to_df()
        if self.start_day != 1:
            df["day"] = self.start_day + df["day"] - 1
            df.index = self.start_day + df.index - 1
        return df


    
