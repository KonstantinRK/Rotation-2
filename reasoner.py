from utils import BaseClass, list_dict, find_cutoff
import numpy as np
from igraph import Graph

class Reasoner:

    def __init__(self):
        pass

    def reason(self, data):
        pass

    def clean_data(self, data):
        pass


class ConceptReasoner2(Reasoner):

    def __init__(self, class_map, concept_hierarchy, is_contradictory_foo=None, max_len=None):
        super(ConceptReasoner2, self).__init__()
        self.class_map = class_map
        self.concept_hierarchy = concept_hierarchy
        self.concept_graph = self.create_graph(concept_hierarchy)
        self.is_contradictory = self._is_contradictory if is_contradictory_foo is None else is_contradictory_foo
        self.value_map = None
        self.max_len = max_len

    def _filter(self, data):
        cutoff = find_cutoff([self.gcv(dp) for dp in data])
        res = []
        for j in sorted(data, key=lambda x: x[1], reverse=True):
            if j[1] >= cutoff:
                res.append(j)
        return res

    @staticmethod
    def gcn(dp):
        if isinstance(dp, str):
            return dp
        else:
            return dp[0]

    @staticmethod
    def gcv(dp):
        return dp[1]

    def get_values(self, dp):
        return [i for i in self.value_map[self.gcn(dp)]]

    def get_entry(self, dp, hierarchy):
        return hierarchy[self.gcn(dp)]

    def get_class(self, dp):
        return self.class_map[self.gcn(dp)]

    def _is_equal(self, dp1, dp2):
        return self.gcn(dp1) == self.gcn(dp2)

    def _is_contradictory(self, dp1, dp2):
        return self.get_class(dp1) == self.get_class(dp2)

    def _set_is_contradictory(self, data):
        result = False
        if self.max_len < len(data):
            return True
        for d1 in data:
            for d2 in data:
                if not self._is_equal(d1, d2):
                    result = self._is_contradictory(d1, d2)
        return result

    def _expand(self, data):
        result = []
        for dp in data:
            res = self.get_subconcepts(dp)
            res.add(dp)
            for e in res:
                self.value_map[self.gcn(e)].add(self.gcv(dp))
            result.append({self.gcn(r) for r in res})
        return result

    def _merge(self, data):
        data = {frozenset(ds) for ds in data}
        results = {frozenset(ds) for ds in data}
        k = len(results)-1
        while len(results) != k:
            k = len(results)
            for ds1 in data:
                for ds2 in data:
                    if self.max_len is None or (len(ds1) <= self.max_len or len(ds2) <= self.max_len):
                        ds3 = ds1.union(ds2)
                        if not self._set_is_contradictory(ds3):
                            results.add(ds3)
        return results

    def get_subconcepts(self, dp):
        sub_concepts = set()
        for p in self.concept_graph.get_all_simple_paths(dp[0]):
            sub_concepts = sub_concepts.union({self.concept_graph.vs["name"][i] for i in p[1:]})
        return sub_concepts

    def _rate_set(self, ds):
        vals = []
        for d in ds:
            vals.append(np.mean(self.get_values(d)))
        return np.mean(vals)

    def _rate_sets(self, data):
        results = []
        for ds in data:
            results.append((ds, self._rate_set(ds)))
        return results

    def reason(self, data):
        results = data[:8]
        self.value_map = {self.gcn(k): {self.gcv(k)} for k in data}
        # results = self._filter(data)
        results = self._expand(results)
        results = self._merge(results)
        results = self._rate_sets(results)
        return results

    def create_graph(self, concept_hierarchy):
        concepts = list(self.class_map.keys())
        # vertices = ["source"] + concepts + ["target"]
        vertices = concepts
        g = Graph(len(vertices), directed=True)
        g.vs["name"] = vertices
        # for i in concepts:
        #     g.add_edge("source", i)
        g = self._add_edges(g, concept_hierarchy)
        return g

    def _add_edges(self, g, hierarchy):
        for k, v in hierarchy.items():
            if v:
                for s in v.keys():
                    g.add_edge(k,s)
                self._add_edges(g, v)
            # else:
            #     g.add_edge(k, "target")
        return g


class ConceptReasoner(Reasoner):

    def __init__(self, class_map, concept_hierarchy, is_contradictory_foo=None, max_len=None):
        super(ConceptReasoner, self).__init__()
        self.class_map = class_map
        self.concept_hierarchy = concept_hierarchy
        self.concept_graph = self.create_graph(concept_hierarchy)
        self.is_contradictory = self._is_contradictory if is_contradictory_foo is None else is_contradictory_foo
        self.value_map = None
        self.max_len = max_len

    # @staticmethod
    # def _filter(data):
    #     vals = np.sort(np.array([k[1] for k in data]))
    #     a = vals.reshape(-1, 1)
    #     kde = KernelDensity(kernel='gaussian', bandwidth=0.01).fit(a)
    #     s = np.linspace(0, 1, num=100)
    #     e = kde.score_samples(s.reshape(-1, 1))
    #     mi, ma = argrelextrema(e, np.less)[0], argrelextrema(e, np.greater)[0]
    #     cutoff = s[mi[-1]]
    #     res = []
    #     for j in sorted(data, key=lambda x: x[1], reverse=True):
    #         if j[1] >= cutoff:
    #             res.append(j)
    #     return res

    def _filter(self, data):
        if len(data) == 0:
            cutoff = 0
        elif len(data) == 1:
            cutoff = self.gcv(data[0])
        else:
            cutoff = find_cutoff([self.gcv(dp) for dp in data])
        res = []
        for j in sorted(data, key=lambda x: x[1], reverse=True):
            if j[1] >= cutoff:
                res.append(j)
        return res

    @staticmethod
    def gcn(dp):
        if isinstance(dp, str):
            return dp
        else:
            return dp[0]

    @staticmethod
    def gcv(dp):
        return dp[1]

    def get_values(self, dp):
        return [i for i in self.value_map[self.gcn(dp)]]

    def get_entry(self, dp, hierarchy):
        return hierarchy[self.gcn(dp)]

    def get_class(self, dp):
        return self.class_map[self.gcn(dp)]

    def _is_equal(self, dp1, dp2):
        return self.gcn(dp1) == self.gcn(dp2)

    def _is_contradictory(self, dp1, dp2):
        return self.get_class(dp1) == self.get_class(dp2)

    def _set_is_contradictory(self, data):
        result = False
        if self.max_len < len(data):
            return True
        for d1 in data:
            for d2 in data:
                if not self._is_equal(d1, d2):
                    result = self._is_contradictory(d1, d2)
        return result

    def get_subconcepts(self, dp):
        sub_concepts = set()
        for p in self.concept_graph.get_all_simple_paths(dp[0]):
            sub_concepts = sub_concepts.union({self.concept_graph.vs["name"][i] for i in p[1:]})
        return sub_concepts

    def expand_concept(self, dp):
        return {self.gcn(dp)}.union(self.get_subconcepts(dp))

    def reason(self, data):
        self.value_map = {self.gcn(d): [self.gcv(d)] for d in data}
        start_vertices = [v for v in self.concept_graph.vs if v.degree(mode="in") == 0]
        self._recompute_values(start_vertices)
        self.value_map = {k: np.mean(v) for k, v in self.value_map.items()}
        data = [(k["name"], self.value_map[k["name"]]) for k in start_vertices]
        result = sorted(data, key=lambda x: x[1], reverse=True)[0]
        return self.expand_concept(result)

    def _recompute_values(self, start_vertices):
        for v in start_vertices:
            self._update_value_map(v)

    def _update_value_map(self, vertex):
        succ = vertex.neighbors(mode="out")
        for s in succ:
            self._update_value_map(s)
            self.value_map[vertex["name"]] += self.value_map[s["name"]]

    def create_graph(self, concept_hierarchy, inverse=False):
        concepts = list(self.class_map.keys())
        # vertices = ["source"] + concepts + ["target"]
        vertices = concepts
        g = Graph(len(vertices), directed=True)
        g.vs["name"] = vertices
        # for i in concepts:
        #     g.add_edge("source", i)
        g = self._add_edges(g, concept_hierarchy, inverse)
        return g

    def get_low_level_concepts(self, data):
        return [d for d in data if self.concept_graph.outdegree(self.gcn(d))==0]

    def _add_edges(self, g, hierarchy, inverse=False):
        for k, v in hierarchy.items():
            if v:
                for s in v.keys():
                    if inverse:
                        g.add_edge(s, k)
                    else:
                        g.add_edge(k, s)
                self._add_edges(g, v, inverse)
        return g


class ShapeConceptReasoner1(ConceptReasoner):

    def __init__(self):
        color_conflict = ["blue", "red", "green"]
        shape_conflict = ["rectangle", "triangle", "circle"]
        composite = [c + "_" + s for c in color_conflict for s in shape_conflict]
        class_map = {k[0]: k[1] for k in ([(i, "color") for i in color_conflict] +
                                         [(i, "shape") for i in shape_conflict] +
                                         [(i, "composite") for i in composite])}
        concept_hierarchy = {k1: {k2: {} for k2 in k1.split("_")} for k1 in composite}
        super().__init__(class_map=class_map, concept_hierarchy=concept_hierarchy, max_len=3)

    def is_color(self, x):
        return self.get_class(x) == "color"

    def is_shape(self, x):
        return self.get_class(x) == "shape"

    def is_comp(self, x):
        return self.get_class(x) == "composite"

    def check_output(self, res):
        if len(res)>3:
            return set()
        else:
            return res

    def get_low_level_concepts(self, data):
        data = super(ShapeConceptReasoner1, self).get_low_level_concepts(data)
        shapes = []
        colors = []
        for d in data:
            if self.is_shape(d):
                shapes.append(d)
            if self.is_color(d):
                colors.append(d)
        return shapes + colors


class ShapeConceptReasoner2(ConceptReasoner):

    def __init__(self):
        color_conflict = ["blue", "red", "green"]
        shape_conflict = ["rectangle", "triangle", "circle"]
        composite = [c + "_" + s for c in color_conflict for s in shape_conflict]
        class_map = {k[0]: k[1] for k in ([(i, "color") for i in color_conflict] +
                                         [(i, "shape") for i in shape_conflict] +
                                         [(i, "composite") for i in composite])}
        concept_hierarchy = {k1: {k2: {} for k2 in k1.split("_")} for k1 in composite}
        super().__init__(class_map=class_map, concept_hierarchy=concept_hierarchy, max_len=3)

    def check_output(self, res):
        if len(res) > 3:
            return set()
        else:
            return res

    def is_color(self, x):
        return self.get_class(x) == "color"

    def is_shape(self, x):
        return self.get_class(x) == "shape"

    def is_comp(self, x):
        return self.get_class(x) == "composite"

    def _separate(self, data):
        shapes = []
        colors = []
        comp = []
        sort = lambda x: sorted(x, key=lambda y: y[1], reverse=True)
        for dp in data:
            if self.is_shape(dp):
                shapes.append(dp)
            elif self.is_color(dp):
                colors.append(dp)
            elif self.is_comp(dp):
                comp.append(dp)
        return sort(shapes), sort(colors), sort(comp)

    def _create_artificial_comp(self, shapes, colors):
        artificial_comp = []
        for s in shapes:
            for c in colors:
                artificial_comp.append(("_".join([self.gcn(c), self.gcn(s)]), np.mean([self.gcv(c), self.gcv(s)])))
        return sorted(artificial_comp, key=lambda y: y[1], reverse=True)

    def select_top(self, data):
        shapes, colors, comp = self._separate(data)
        artificial_comp = self._create_artificial_comp(shapes, colors)
        top_comp = self._filter(comp)
        top_a_comp = self._filter(artificial_comp)
        res = [self.resolve_conflict(top_comp, artificial_comp), self.resolve_conflict(top_a_comp, comp)]
        return sorted(res, key=lambda y: y[1], reverse=True)[0]

    def _create_value_map(self, data):
        return {self.gcn(d): self.gcv(d) for d in data}

    def resolve_conflict(self, prime_data, second_data):
        second_data_map = self._create_value_map(second_data)
        prime_data_res = sorted(prime_data, key=lambda x: second_data_map[self.gcn(x)], reverse=True)
        return prime_data_res[0]

    def reason(self, data):
        self.value_map = {self.gcn(dp): self.gcv(dp) for dp in data}
        result = self.select_top(data)
        return self.expand_concept(result)

    def get_low_level_concepts(self, data):
        data = super(ShapeConceptReasoner2, self).get_low_level_concepts(data)
        shapes = []
        colors = []
        for d in data:
            if self.is_shape(d):
                shapes.append(d)
            if self.is_color(d):
                colors.append(d)
        return shapes + colors


class ShapeConceptReasoner3(ShapeConceptReasoner2):

    def select_top(self, data):

        shapes, color, comp = self._separate(data)
        artificial_comp = []
        for s in shapes:
            for c in color:
                artificial_comp.append(("_".join([self.gcn(c), self.gcn(s)]), np.mean([self.gcv(c), self.gcv(s)])))
        artificial_comp = sorted(artificial_comp, key=lambda y: y[1], reverse=True)
        top_comp = self._filter(comp)
        res = self.resolve_conflict(top_comp, artificial_comp)
        return res


class ShapeConceptReasoner4(ShapeConceptReasoner2):

    def reason(self, data):
        shapes, color, comp = self._separate(data)
        res = []
        max_val = self.gcv(comp[0])
        for c in comp:
            if self.gcv(c) < max_val:
                break
            res += self.expand_concept(c)
        return self.check_output(set(res))


class ShapeConceptReasoner5(ShapeConceptReasoner2):

    def reason(self, data):
        shapes, colors, comp = self._separate(data)
        res = []
        comp = self._create_artificial_comp(shapes, colors)
        max_val = self.gcv(comp[0])
        for c in comp:
            if self.gcv(c) < max_val:
                break
            res += self.expand_concept(c)
        # print(res)
        # print("-" * 100)
        # ch = sorted(data, key=lambda x: x[1], reverse=True)
        # for i in ch:
        #     if self.is_color(i):
        #         print(i)
        # print("")
        # for i in ch:
        #     if self.is_shape(i):
        #         print(i)
        # print("")
        # for i in ch:
        #     if self.is_comp(i):
        #         print(i)
        # print("-"*100)
        # print("")
        return self.check_output(set(res))


class ShapeConceptReasoner6(ShapeConceptReasoner1):

    def reason(self, data):
        self.value_map = {self.gcn(d): self.gcv(d) for d in data}
        start_vertices = [v for v in self.concept_graph.vs if v.degree(mode="in") == 0]
        self._recompute_values(start_vertices)
        data = [(k["name"], self.value_map[k["name"]]) for k in start_vertices]
        result = sorted(data, key=lambda x: x[1], reverse=True)[0]
        res = self.expand_concept(result)
        return self.check_output(set(res))

    def _recompute_values(self, start_vertices):
        for v in start_vertices:
            self._update_value_map(v)

    def _update_value_map(self, vertex):
        succ = vertex.neighbors(mode="out")
        if len(succ)==0:
            return self.value_map[vertex["name"]]
        succ_vals = []
        for s in succ:
            succ_vals.append(self._update_value_map(s))
        self.value_map[vertex["name"]] = np.average([self.value_map[vertex["name"]], np.average(succ_vals)])
        return self.value_map[vertex["name"]]


class ShapeConceptReasoner7(ShapeConceptReasoner2):

    def reason(self, data):
        res = self._filter(data)
        res = [self.gcn(r) for r in res]
        return self.check_output(set(res))
