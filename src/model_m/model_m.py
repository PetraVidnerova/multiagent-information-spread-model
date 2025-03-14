import os
import pickle

import numpy as np
from utils.config_utils import ConfigFile

from graphs.graph_gen import GraphGenerator, CSVGraphGenerator, RandomSingleGraphGenerator
from graphs.light import LightGraph
from models.model_zoo import model_zoo
from models.states import STATES

import logging


def load_model_from_config(cf, model_random_seed, preloaded_graph=None, hyperparams=None, policy_params=None):

    # load model hyperparameters
    model_params = cf.section_as_dict("MODEL")
    if hyperparams is not None:
        model_params = {**model_params, **hyperparams}

    # load graph as described in config file
    if preloaded_graph is not None:
        graph = preloaded_graph
    else:
        graph = load_graph(cf)

    #  policy
    policy = _load_policy_function(cf, policy_params=policy_params)

    # sceanario
    scenario = cf.section_as_dict("SCENARIO")
    scenario = scenario["closed"] if scenario else None

    # model type
    model_type = cf.section_as_dict("TASK").get(
        "model", "SimulationDrivenModel")

    model = ModelM(graph,
                   policy,
                   model_params,
                   scenario,
                   random_seed=model_random_seed,
                   model_type=model_type
                   )
    return model


class ModelM():

    def __init__(self,
                 graph,
                 policy,
                 model_params: dict = None,
                 scenario: list = None,
                 model_type: str = "SimulationDrivenModel",
                 random_seed: int = 42):

        # self.random_seed = 42

        # original state
        self.start_graph = graph

        # scenario (list of closed layers)
        self.scenario = scenario

        self.model_type = model_type
        self.model_params = model_params
        self.random_seed = random_seed
        self.policy = policy
        self.policy_object = None

        self.model = None
        self.ready = False

        self.graph = None
        self.A = None

    def setup(self):
        logging.info("model setup")
        # working copy of graph and matrix
        if self.graph is not None:
            del self.graph
        if self.A is not None:
            del self.A
        if self.model is not None:
            del self.model

        if self.policy_object is not None:
            self.policy_object.graph = None
            self.policy_object.model = None
            del self.policy_object

        self.graph = self.start_graph.copy()
        self.A = self.init_matrix()

        # model
        Model = model_zoo[self.model_type]
        self.model = Model(self.A,
                           **self.model_params,
                           random_seed=self.random_seed)
        if self.policy[0] is not None:
            self.policy_object = self.policy[0](
                self.graph, self.model, **self.policy[1])
        else:
            self.policy_object = None
        self.model.set_periodic_update(self.policy_object)
        self.ready = True

    def duplicate(self, random_seed=None, hyperparams=None, policy_params=None):
        #        pritn("DBD duplicate")

        if self.ready:
            raise NotImplementedError("We duplicate only newbie models")

        if policy_params is not None:
            policy_func, policy_setup = self.policy
            policy_setup = {**policy_setup, **policy_params}
            policy = policy_func, policy_setup
        else:
            policy = self.policy

        twin = ModelM(
            self.start_graph,
            policy,
            self.model_params if hyperparams is None else dict(
                self.model_params, **hyperparams),
            self.scenario,
            random_seed=self.random_seed if random_seed is None else random_seed,
            model_type=self.model_type
        )
        twin.graph = twin.start_graph.copy()
        twin.A = twin.init_matrix()
        twin.ready = False
        return twin

    def set_model_params(self, model_params: dict):
        self.model.setup_model_params(model_params)

    def run(self, *args, **kwargs):
        if not self.ready:
            self.setup()
        self.model.run(*args, **kwargs)

    def reset(self, random_seed=None):
        #        print("DBD reset")
        if not self.ready:
            self.setup()
        else:
            del self.graph
            self.graph = self.start_graph.copy()
            del self.A
            self.A = self.init_matrix()
            self.model.update_graph(self.graph)

            if self.policy_object is not None:
                del self.policy_object
            if self.policy[0] is not None:
                self.policy_object = self.policy[0](
                    self.graph, self.model, **self.policy[1])
            else:
                self.policy_object = None
            self.model.set_periodic_update(self.policy_object)

        self.model.inicialization()

        # random_seed has to be setup AFTER inicialization and BEFORE states_and_counts_init !
        if random_seed:
            self.model.set_seed(random_seed)
            logging.debug(f"seed changed to {self.model.random_seed}")

        self.model.setup_series_and_time_keeping()
        self.model.states_and_counts_init(ext_nodes=self.model.num_ext_nodes,
                                          ext_code=STATES.EXT)

    def get_results(self,
                    states):
        if type(states) == list:
            return [self.model.get_state_count(s) for s in states]
        else:
            return self.model.get_state_count(states)

    def get_df(self):
        model_df = self.model.to_df()
        if self.policy_object is not None:
            policy_df = self.policy_object.to_df()
        else:
            policy_df = None

        df = model_df.merge(policy_df, on="T",
                            how="outer", suffixes=("", "_policy")) if policy_df is not None else model_df
        return df

    def save_history(self, file_or_filename):
        model_df = self.model.to_df()
        if self.policy_object is not None:
            policy_df = self.policy_object.to_df()
        else:
            policy_df = None

        df = model_df.merge(policy_df, on="T",
                            how="outer", suffixes=("", "_policy")) if policy_df is not None else model_df

#        cols = df.columns.tolist()
#        print(cols)
#        cols.insert(0, cols.pop(cols.index('T')))
#        df = df.reindex(columns= cols)

        df.to_csv(file_or_filename)
        print(df)

    def save_node_states(self, filename):
        self.model.save_node_states(filename)

    def init_matrix(self):
        if isinstance(self.graph, LightGraph):
            #            raise NotImplementedError(
            #                "LighGraph not  supported at the moment, waits for fixes.")
            return self.graph

        if isinstance(self.graph, RandomSingleGraphGenerator):
            if self.scenario:
                raise NotImplementedError(
                    "RandomGraphGenerator does not support layers.")
            return grahp.G

        # this is what we currently used
        if isinstance(self.graph, GraphGenerator):
            if self.scenario:
                self.graph.close_layers(self.scenario)
            return self.graph.final_adjacency_matrix()

        raise TypeError("Unknown type of graph")


def load_graph(cf: ConfigFile):
    logging.info("Load graph.")

    num_nodes = cf.section_as_dict("TASK").get("num_nodes", None)

    cf_graph = cf.section_as_dict("GRAPH")

    graph_type = cf_graph["type"]
    filename = cf_graph.get("file", None)
    nodes = cf_graph.get("nodes", "nodes.csv")
    edges = cf_graph.get("edges", "edges.csv")
    layers = cf_graph.get("layers", "etypes.csv")
    externals = cf_graph.get("externals", None)
    quarantine = cf_graph.get("quarantine", None)
    layer_groups = cf_graph.get("layer_groups", None)

    if filename is not None and os.path.exists(filename):
        graph_type = 'pickle'

    if graph_type == "csv":
        raise NotImplementedError(
            "Sorry. Graph_type 'csv' is not supported anymore. Use light graph.")
        # g = CSVGraphGenerator(path_to_nodes=nodes,
        #                       path_to_external=externals,
        #                       path_to_edges=edges,
        #                       path_to_layers=layers,
        #                       path_to_quarantine=quarantine)

#    if graph_name == "csv_light":
#        return LightGraph(path_to_nodes=nodes, path_to_edges=edges, path_to_layers=layers)

    elif graph_type == "light":
        g = LightGraph()
        g.read_csv(path_to_nodes=nodes,
                   path_to_external=externals,
                   path_to_edges=edges,
                   path_to_layers=layers,
                   path_to_quarantine=quarantine,
                   path_to_layer_groups=layer_groups)

    elif graph_type == "random":
        raise NotImplementedError(
            "Sorry. Graph_type 'random' is not supported now. Use light graph and stay tuned.")
        # g = RandomGraphGenerator()

    elif graph_type == "pickle":
        with open(filename, "rb") as f:
            g = pickle.load(f)
            if isinstance(g, GraphGenerator):
                if g.A_valid:
                    print("Wow, matrix A is ready.")
            else:
                assert isinstance(g, LightGraph), f"Something weird ({type(g)}) was loaded."
    else:
        raise ValueError(f"Graph {graph_type} not available.")

    if graph_type != "pickle" and filename is not None:
        save_graph(filename, g)
    return g


def save_graph(filename, graph):
    with open(filename, 'wb') as f:
        pickle.dump(graph, f, protocol=4)


def _load_policy_function(cf: ConfigFile, policy_params=None):
    policy_cfg = cf.section_as_dict("POLICY")

    policy_name = policy_cfg.get("name", None)
    if policy_name is None:
        return None, None

    if "filename" in policy_cfg:
        PolicyClass = getattr(
            __import__(
                "policies."+policy_cfg["filename"],
                globals(), locals(),
                [policy_name],
                0
            ),
            policy_name
        )
        setup = cf.section_as_dict("POLICY_SETUP")
        if policy_params is not None:
            setup = {**setup, **policy_params}

        return PolicyClass, setup
    else:
        print("Warning: NO POLICY IN CFG")
        print(policy_cfg)
        raise ValueError("Unknown policy.")
