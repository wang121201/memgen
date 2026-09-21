#ifndef HYFISS_PRIVATE_CACHE_CHECKPOINT_H
#define HYFISS_PRIVATE_CACHE_CHECKPOINT_H

// Private generator implementation, included after Options/SectorLruCache.
// Only kernel-boundary cache state is resumed; output streams start afresh.
constexpr const char *kCacheCheckpointMagic = "HYFISS_CACHE_CHECKPOINT_V4\n";

std::string cache_checkpoint_contract(const Options &o) {
  std::ostringstream s;
  s << "mapping=fallback_quotient_v2;explicit_accelsim_v1;local_owner_v3\n";
  s << "data_validity=known_bytes_union_v1;pending_read=independent_v1;l1_read=ldg_strong_gpu_bypass_v1\n";
  s << "writeback=dirty_sector_runs_v1;drain_budget=global_remaining_v2\n";
#define CACHE_FIELD(name) s << #name << '=' << o.name << '\n'
  CACHE_FIELD(num_sms); CACHE_FIELD(num_partitions);
  CACHE_FIELD(num_memory_channels); CACHE_FIELD(num_sub_partitions_per_channel);
  CACHE_FIELD(partition_index_bit); CACHE_FIELD(memory_partition_indexing);
  CACHE_FIELD(mem_address_mask); CACHE_FIELD(mem_addr_mapping);
  CACHE_FIELD(sector_size); CACHE_FIELD(l1_size_bytes); CACHE_FIELD(l2_size_bytes);
  CACHE_FIELD(l1_line_size); CACHE_FIELD(l2_line_size);
  CACHE_FIELD(l1_assoc); CACHE_FIELD(l2_assoc);
  s << "l1_set_index=" << set_index_name(o.l1_set_index) << '\n'
    << "l2_set_index=" << set_index_name(o.l2_set_index) << '\n';
  CACHE_FIELD(l1_fill_latency); CACHE_FIELD(l2_fill_latency);
  CACHE_FIELD(l1_store_policy); CACHE_FIELD(dram_store_policy);
  CACHE_FIELD(write_sector_policy); CACHE_FIELD(preserve_l1); CACHE_FIELD(preserve_l2);
  CACHE_FIELD(flush_l2_on_reset); CACHE_FIELD(l2_dirty_drain); CACHE_FIELD(l2_streaming_fill);
  CACHE_FIELD(l2_dirty_drain_latency); CACHE_FIELD(l2_dirty_drain_max_sectors_per_kernel);
  CACHE_FIELD(l2_dirty_drain_high_watermark_sectors); CACHE_FIELD(l2_dirty_drain_target_sectors);
  CACHE_FIELD(include_local); CACHE_FIELD(sort_by_timestamp); CACHE_FIELD(monotonic_sm);
  CACHE_FIELD(issue_interval); CACHE_FIELD(kernel_gap); CACHE_FIELD(input_format);
#undef CACHE_FIELD
  // Exact metadata bytes and canonical source directories prevent accidentally
  // continuing another capture. Trace-byte SHA verification remains a runner
  // responsibility; this is not an adversarial filesystem snapshot.
  for (const auto &dir : {o.configs_dir, o.memory_dir})
    s << "source_dir=" << fs::weakly_canonical(dir).generic_string() << '\n';
  for (const auto &p : {o.configs_dir / "app.config", o.configs_dir / "issue.config",
                        o.configs_dir / "function.config", o.configs_dir / "capture_receipt.json",
                        o.semantic_file}) {
    if (p.empty() || !fs::exists(p)) { s << "absent=" << p.generic_string() << '\n'; continue; }
    const auto size = fs::file_size(p);
    if (size > 64 * 1024 * 1024)
      throw std::runtime_error("checkpoint metadata file exceeds 64 MiB bound");
    std::ifstream in(p, std::ios::binary);
    if (!in) throw std::runtime_error("cannot read checkpoint metadata: " + p.string());
    std::string bytes(size, '\0');
    in.read(bytes.data(), static_cast<std::streamsize>(size));
    if (!in || in.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("checkpoint metadata changed while reading");
    s << "file=" << fs::weakly_canonical(p).generic_string() << '\n' << size << '\n';
    s.write(bytes.data(), static_cast<std::streamsize>(size));
    s << '\n';
  }
  return s.str();
}

uint64_t cache_checkpoint_size_bound(const Options &o, size_t contract_size) {
  const uint64_t l1_lines = uint64_t(std::max(1u, o.num_sms)) *
      std::max<uint64_t>(o.l1_assoc, o.l1_size_bytes / std::max(1u, o.l1_line_size));
  const uint64_t l2_lines = uint64_t(std::max(1u, o.num_partitions)) *
      std::max<uint64_t>(o.l2_assoc, uint64_t(o.l2_size_bytes) /
                         std::max(1u, o.num_partitions) / std::max(1u, o.l2_line_size));
  // Worst serialized line: tag, sector and byte masks, 32 reserved/dirty timestamps,
  // per-set envelope; additional space covers cache headers and the contract.
  if (l1_lines + l2_lines > (UINT64_MAX - contract_size - 1024 * 1024) / 768)
    throw std::runtime_error("checkpoint geometry exceeds size bound");
  return (l1_lines + l2_lines) * 768 + contract_size + 1024 * 1024;
}

void save_cache_checkpoint(const fs::path &path, uint64_t kernel_base,
    size_t position, int next_kernel_id, const std::string &contract,
    const std::vector<SectorLruCache> &l1, const std::vector<SectorLruCache> &l2) {
  if (fs::exists(path))
    throw std::runtime_error("checkpoint path exists; preserve it and use a fresh directory");
  std::ostringstream payload(std::ios::binary);
  write_pod(payload, uint64_t(contract.size())); payload.write(contract.data(), contract.size());
  write_pod(payload, uint64_t(position)); write_pod(payload, int64_t(next_kernel_id));
  write_pod(payload, kernel_base);
  write_pod(payload, uint64_t(l1.size()));
  for (const auto &cache : l1) cache.save(payload);
  write_pod(payload, uint64_t(l2.size()));
  for (const auto &cache : l2) cache.save(payload);
  const auto body = payload.str();
  boost::crc_32_type checksum; checksum.process_bytes(body.data(), body.size());
  fs::create_directories(path.parent_path());
  // Directory creation claims this staging name atomically. Publish with a
  // no-replace hard link only after close succeeds; retain staging evidence.
  // Both names then refer to the same data, not a second file-content copy.
  const fs::path pending = path.string() + ".pending";
  if (!fs::create_directory(pending))
    throw std::runtime_error("checkpoint staging exists; preserve it and use a fresh directory");
  const fs::path staged = pending / "payload.bin";
  std::ofstream os(staged, std::ios::binary);
  if (!os) throw std::runtime_error("cannot create checkpoint: " + path.string());
  os << kCacheCheckpointMagic;
  write_pod(os, uint64_t(body.size())); write_pod(os, uint32_t(checksum.checksum()));
  os.write(body.data(), static_cast<std::streamsize>(body.size()));
  os.flush(); os.close();
  if (!os) throw std::runtime_error("failed writing checkpoint: " + path.string());
  fs::create_hard_link(staged, path); // Fails if another writer created path.
}

uint64_t load_cache_checkpoint(const fs::path &path, const Options &o,
    const std::string &contract, int expected_next_kernel,
    std::vector<SectorLruCache> &l1, std::vector<SectorLruCache> &l2) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("cannot open checkpoint: " + path.string());
  const std::string expected = kCacheCheckpointMagic;
  std::string magic(expected.size(), '\0');
  file.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (magic != expected)
    throw std::runtime_error("unsupported checkpoint: V4 with byte-valid L1/L2 and independent pending reads is required");
  const auto bytes = read_pod<uint64_t>(file, "checkpoint payload length");
  const auto recorded_crc = read_pod<uint32_t>(file, "checkpoint payload CRC32");
  if (bytes > cache_checkpoint_size_bound(o, contract.size()) ||
      fs::file_size(path) != expected.size() + sizeof(uint64_t) + sizeof(uint32_t) + bytes)
    throw std::runtime_error("checkpoint payload size mismatch");
  std::string body(static_cast<size_t>(bytes), '\0');
  file.read(body.data(), static_cast<std::streamsize>(body.size()));
  if (!file || file.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("truncated or trailing checkpoint data");
  boost::crc_32_type checksum; checksum.process_bytes(body.data(), body.size());
  if (checksum.checksum() != recorded_crc)
    throw std::runtime_error("checkpoint payload checksum mismatch");
  std::istringstream is(body, std::ios::binary);
  const auto contract_size = read_pod<uint64_t>(is, "checkpoint contract size");
  if (contract_size != contract.size()) throw std::runtime_error("checkpoint contract mismatch");
  std::string recorded_contract(contract.size(), '\0');
  is.read(recorded_contract.data(), recorded_contract.size());
  if (!is || recorded_contract != contract) throw std::runtime_error("checkpoint contract mismatch");
  (void)read_pod<uint64_t>(is, "checkpoint position");
  const auto next = read_pod<int64_t>(is, "checkpoint next kernel");
  if (next < 0 || next != expected_next_kernel)
    throw std::runtime_error("checkpoint next kernel mismatch or terminal checkpoint");
  const auto kernel_base = read_pod<uint64_t>(is, "checkpoint kernel_base");
  auto restored_l1 = l1; auto restored_l2 = l2;
  if (read_pod<uint64_t>(is, "checkpoint L1 count") != restored_l1.size())
    throw std::runtime_error("checkpoint SM count mismatch");
  for (auto &cache : restored_l1) cache.load(is, o.sector_size);
  if (read_pod<uint64_t>(is, "checkpoint L2 count") != restored_l2.size())
    throw std::runtime_error("checkpoint L2 partition count mismatch");
  for (size_t p = 0; p < restored_l2.size(); ++p) {
    restored_l2[p].load(is, o.sector_size, [&](uint64_t address) {
      if (dram_partition_index(address, o) != p)
        throw std::runtime_error("checkpoint tag in wrong L2 partition");
      return l2_cache_index_addr(address, o);
    });
  }
  if (is.peek() != std::char_traits<char>::eof())
    throw std::runtime_error("trailing checkpoint cache payload");
  l1.swap(restored_l1); l2.swap(restored_l2);
  return kernel_base;
}
#endif
