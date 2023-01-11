use std::ops::Range;

pub type NodeKey = u32;
pub type ChildRelativeKey = u32;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DFSCommand {
    Continue,
    Ascend,
    End,
}

pub trait NodeGraph {
    type NodeData;

    fn node_count(&self) -> usize;

    fn nodes(&self) -> Range<NodeKey>;
    fn children(&self, this: NodeKey) -> Range<ChildRelativeKey>;
    fn get_child(&self, this: NodeKey, child: ChildRelativeKey) -> NodeKey;

    fn get_node_data(&self, this: NodeKey) -> &Self::NodeData;
    fn get_node_data_mut(&mut self, this: NodeKey) -> &mut Self::NodeData;

    fn dfs_visit<F: FnMut(NodeKey) -> DFSCommand>(&self, start: NodeKey, mut fun: F) {
        dfs_visit_inner(self, start, &mut fun);
    }
}

fn dfs_visit_inner<T: NodeGraph + ?Sized, F: FnMut(NodeKey) -> DFSCommand>(
    graph: &T,
    root: NodeKey,
    fun: &mut F,
) -> DFSCommand {
    match fun(root) {
        DFSCommand::Continue => {}
        DFSCommand::End => return DFSCommand::End,
        DFSCommand::Ascend => return DFSCommand::Ascend,
    }

    for c in graph.children(root) {
        let node = graph.get_child(root, c);
        match dfs_visit_inner(graph, root, fun) {
            DFSCommand::Continue => {}
            DFSCommand::Ascend => return DFSCommand::Continue,
            DFSCommand::End => return DFSCommand::End,
        }
    }

    DFSCommand::Continue
}

#[derive(Default, Clone)]
pub struct ImmutableGraph {
    nodes: Vec<(u32, u32)>,
    children: Vec<u32>,
    dummy_unit: (),
}

impl ImmutableGraph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn get_children(&self, node: NodeKey) -> &[NodeKey] {
        let node = self.nodes[node as usize];
        &self.children[node.0 as usize..node.1 as usize]
    }
}

impl NodeGraph for ImmutableGraph {
    type NodeData = ();

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn nodes(&self) -> Range<NodeKey> {
        0..self.nodes.len() as u32
    }

    fn children(&self, this: NodeKey) -> Range<ChildRelativeKey> {
        let node = self.nodes[this as usize];
        0..node.1 - node.0
    }

    fn get_child(&self, this: NodeKey, child: ChildRelativeKey) -> NodeKey {
        let node = self.nodes[this as usize];
        self.children[(node.0 + child) as usize]
    }

    fn get_node_data(&self, this: NodeKey) -> &Self::NodeData {
        &()
    }

    fn get_node_data_mut(&mut self, this: NodeKey) -> &mut Self::NodeData {
        &mut self.dummy_unit
    }
}

pub fn reverse_edges<D, G: NodeGraph<NodeData = D>>(graph: &G) -> ImmutableGraph {
    let mut into = Default::default();
    reverse_edges_into(graph, &mut into);
    into
}

pub fn reverse_edges_into<D, G: NodeGraph<NodeData = D>>(graph: &G, into: &mut ImmutableGraph) {
    // collect the parents of nodes
    // 1. iterate over all nodes, extract their dependencies, and from this count the number of dependees (kept in the dependees_start field)
    // 2. prefix sum over the nodes, allocating space in a single vector, offset into which is stored in dependees_start,
    //    dependees_end now serves as the counter of pushed dependees (the previous count in dependees_start is not needed anymore)
    // 3. iterate over all nodes, extract their dependencies, and store them in the allocated space, dependees_end now points at the end

    let ImmutableGraph {
        nodes, children, ..
    } = into;
    nodes.resize(graph.node_count(), (0, 0));

    // 1.
    for node in graph.nodes() {
        for c in graph.children(node) {
            let child = graph.get_child(node, c);
            nodes[child as usize].0 += 1;
        }
    }

    // 2.
    let mut offset = 0;
    for node in graph.nodes() {
        // dependees_start currently stores the count of dependees
        let node = &mut nodes[node as usize];
        let end = offset + node.0;

        // start and end are the same on purpose
        *node = (offset, offset);

        offset = end;
    }

    children.resize(offset as usize, 0);

    // 3.
    for p in graph.nodes() {
        for c in graph.children(p) {
            let child = graph.get_child(p, c);

            let offset = &mut nodes[child as usize].1;
            children[*offset as usize] = p;
            *offset += 1;
        }
    }
}
