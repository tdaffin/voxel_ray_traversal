#[derive(Debug)]
pub enum VoxelJobMessage {
    Finished { generation: u64, index: usize, data: Vec<u128> },
    Progress { generation: u64, index: usize, done: usize, total: usize },
    Cancelled { generation: u64 },
}
