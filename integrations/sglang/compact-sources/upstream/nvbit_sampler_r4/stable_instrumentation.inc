// Mechanically extracted from the pinned original instrument_function_if_needed.
// Sink legacy text sidecars in a no-op stream; new static rows travel in pipe.
using namespace std;
struct SGNullStream {template<class T>SGNullStream&operator<<(const T&){return *this;}explicit operator bool()const{return true;}};
static SGNullStream function_config_fp,transfer_instructions_fp,load_mask_instructions_fp;
static uint64_t transfer_static_records=0,load_mask_static_records=0;
static uint32_t instr_begin_interval=0,instr_end_interval=UINT32_MAX;
// These names retain the original classifier's stable-packet admission guard;
// the new transport does not create a MEMC file.
enum SGTraceMode{TRACE_LANE};enum SGTraceFormat{LANE_FORMAT_MEMC};
static const SGTraceMode fast_trace_mode=TRACE_LANE;
static const SGTraceFormat lane_trace_format=LANE_FORMAT_MEMC;
static std::unordered_set<uint64_t> already_instrumented;
static std::map<std::string,int> opcode_to_id_map;
static std::map<int,std::string> id_to_opcode_map;
void instrument_function_if_needed(CUcontext ctx, CUfunction func) {
  // Global channel/config state supports one context. Never reuse an SM or
  // function identity across contexts, including sequential context teardown.
  CUcontext current = nullptr;
  CUdevice device; int major = 0, minor = 0;
  CUDA_SAFECALL(cuCtxGetCurrent(&current));
  if (!ctx || current != ctx || (capture_context && capture_context != ctx))
    throw runtime_error("capture requires one actual CUDA context");
  capture_context = ctx;
  CUDA_SAFECALL(cuCtxGetDevice(&device));
  CUDA_SAFECALL(cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device));
  CUDA_SAFECALL(cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device));
  transfer_sm89 = major == 8 && minor == 9;
  vector<CUfunction> related_functions = nvbit_get_related_functions(ctx, func);
  related_functions.push_back(func);

  for (auto f : related_functions) {
    const uint64_t actual_function_id=sgobs::describe(ctx,f).id;
    if (!already_instrumented.insert(actual_function_id).second) continue;
    if(actual_function_id>UINT32_MAX)throw runtime_error("function ID width");
    const uint32_t function_id=uint32_t(actual_function_id);
    function_config_fp << function_id << "\t" << nvbit_get_func_name(ctx, f, true) << "\n";
    if (!function_config_fp)
      throw runtime_error("cannot persist function identity");

    const vector<Instr *> &instrs = nvbit_get_instrs(ctx, f);
    if(instrs.empty())throw runtime_error("entry proof requires first original instruction");
    // Install on first instrumentation even when f is currently only related;
    // the device function-id guard selects the true entry on every launch.
    auto entry=instrs.front();
    nvbit_insert_call(entry,"instrument_cta_entry",IPOINT_BEFORE);
    nvbit_add_call_arg_const_val32(entry,function_id);
    nvbit_add_call_arg_const_val64(entry,(uint64_t)&sg_sample_entry_function);
    nvbit_add_call_arg_const_val64(entry,(uint64_t)&sg_sample_cta_count);
    nvbit_add_call_arg_const_val64(entry,(uint64_t)sg_sample_cta_ids);
    nvbit_add_call_arg_const_val64(entry,(uint64_t)sg_entry_calls);
    nvbit_add_call_arg_const_val64(entry,(uint64_t)sg_entry_masks);
    uint32_t cnt = 0;
    for (auto instr : instrs) {
      const auto memory_space=instr->getMemorySpace();
      if(memory_space!=InstrType::MemorySpace::NONE &&
         memory_space!=InstrType::MemorySpace::GLOBAL && memory_space!=InstrType::MemorySpace::LOCAL &&
         memory_space!=InstrType::MemorySpace::GENERIC && memory_space!=InstrType::MemorySpace::GLOBAL_TO_SHARED)
        ++sgsample::omitted_spaces[function_id][InstrType::MemorySpaceStr[int(memory_space)]];

      if (!strcmp(instr->getOpcode(), "CALL.REL.NOINC")) {
        cnt++;
        continue;
      }
      if (cnt < instr_begin_interval || cnt >= instr_end_interval) {
        cnt++;
        continue;
      }

      int is_glob_loc = 0;
      int mref_id = 0;
      if (instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL ||
          instr->getMemorySpace() == InstrType::MemorySpace::LOCAL ||
          instr->getMemorySpace() == InstrType::MemorySpace::GENERIC ||
          instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL_TO_SHARED) {
        is_glob_loc = 1;
      }

      for (int i = 0; i < instr->getNumOperands(); i++) {
        const InstrType::operand_t *op = instr->getOperand(i);
        if (op->type == InstrType::OperandType::MREF && is_glob_loc)
          mref_id++;
      }

      if (mref_id > 2) {
        cerr << "unsupported memory-reference count " << mref_id
             << " for opcode " << instr->getOpcode() << endl;
        exit(2);
      }

      if(is_glob_loc && mref_id==0)
        ++sgsample::omitted_spaces[function_id]["UNINSTRUMENTED_MREF_ZERO_"+string(InstrType::MemorySpaceStr[int(memory_space)])];
      if (!is_glob_loc || mref_id == 0) {
        cnt++;
        continue;
      }

      int space_hint = FAST_SPACE_UNKNOWN;
      // NVBit 1.7.6 exposes space per instruction, not per MREF. Do not label
      // both operands of a transfer as global or run generic predicates on a
      // shared offset. Keep both references unresolved until roles are proved.
      if (mref_id == 1) {
        if (instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL)
          space_hint = FAST_SPACE_GLOBAL;
        else if (instr->getMemorySpace() == InstrType::MemorySpace::LOCAL)
          space_hint = FAST_SPACE_LOCAL;
        else if (instr->getMemorySpace() == InstrType::MemorySpace::GENERIC)
          space_hint = FAST_SPACE_GENERIC;
      }

      unsigned transfer_width = 0, transfer_policy = 0, load_mask_width = 0;
      int source_predicate_type = 0, source_predicate_number = 7, source_predicate_not = 0;
      if (instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL_TO_SHARED) {
        const string op(instr->getOpcode());
        if (op == "LDGSTS.E") transfer_width = 4;
        else if (op == "LDGSTS.E.64") transfer_width = 8;
        else if (op == "LDGSTS.E.128") transfer_width = 16;
        else if (op == "LDGSTS.E.BYPASS.128") { transfer_width = 16; transfer_policy = 1; }
        else if (op == "LDGSTS.E.LTC128B") transfer_width = 4;
        else if (op == "LDGSTS.E.LTC128B.64") transfer_width = 8;
        else if (op == "LDGSTS.E.LTC128B.128") transfer_width = 16;
        else if (op == "LDGSTS.E.BYPASS.LTC128B.128" ||
                 op == "LDGSTS.E.BYPASS.LTC128B.128.CONSTANT") {
          // Retain the exact hint opcode in the sealed static table. This
          // records source operands; it does not admit a cache-policy model.
          transfer_width = 16; transfer_policy = 1;
        }
        else throw runtime_error("unsupported async-copy opcode: " + op);
        if (!transfer_sm89 || NVBIT_VERSION != string("1.7.6") ||
            fast_trace_mode != TRACE_LANE || lane_trace_format != LANE_FORMAT_MEMC ||
            mref_id != 2 || (instr->getNumOperands() != 2 && instr->getNumOperands() != 3) ||
            instr->getOperand(0)->type != InstrType::OperandType::MREF ||
            instr->getOperand(1)->type != InstrType::OperandType::MREF)
          throw runtime_error("unsupported async-copy target/form/capture mode");
        if (instr->getNumOperands() == 3) {
          const auto *control = instr->getOperand(2);
          // Only lane predicates have independent source-control witnesses.
          if (control->type != InstrType::OperandType::PRED)
            throw runtime_error("unvalidated async-copy source predicate type");
          source_predicate_type = 1;
          source_predicate_number = control->u.pred.num;
          source_predicate_not = control->is_not ? 1 : 0;
          if (!source_predicate_type || source_predicate_number < 0 || source_predicate_number > 7 ||
              control->is_neg || control->is_abs)
            throw runtime_error("unsupported async-copy source control");
        }
        space_hint = FAST_SPACE_ASYNC_G2S;
        string sass(instr->getSass());
        if (sass.find('\t') != string::npos || sass.find('\n') != string::npos)
          throw runtime_error("invalid async-copy instruction metadata text");
        transfer_instructions_fp << function_id << '\t' << instr->getOffset() << '\t' << op << '\t'
          << transfer_width << '\t' << transfer_policy << '\t' << source_predicate_type << '\t'
          << source_predicate_number << '\t' << source_predicate_not << '\t' << instr->getSize() << '\t' << sass << '\n';
        if (!transfer_instructions_fp) throw runtime_error("cannot persist transfer instruction metadata");
        ++transfer_static_records;
      }

      // Candidate source control; interpretation requires dynamic validation.
      // Classify every global LDG, including ordinary two-operand forms.
      if (instr->getMemorySpace() == InstrType::MemorySpace::GLOBAL &&
          string(instr->getOpcode()).rfind("LDG.", 0) == 0) {
        const string op(instr->getOpcode());
        if (!transfer_sm89 || NVBIT_VERSION != string("1.7.6") ||
            fast_trace_mode != TRACE_LANE || lane_trace_format != LANE_FORMAT_MEMC ||
            mref_id != 1 || (instr->getNumOperands() != 2 && instr->getNumOperands() != 3) ||
            instr->getOperand(0)->type != InstrType::OperandType::REG ||
            instr->getOperand(1)->type != InstrType::OperandType::MREF)
          throw runtime_error("unclassified LDG operand form: " + op);
        if (instr->getNumOperands() == 3) {
          // Exact additional form observed in the sealed native r6 SASS.
          // Preserve the raw source predicate as a candidate witness; this
          // does not qualify its semantics or convert it to ordinary READ.
          const unsigned candidate_width = op == "LDG.E.LTC128B.64.STRONG.GPU" ? 8u :
              (op == "LDG.E.LTC128B.128.CONSTANT" ||
               op == "LDG.E.LTC128B.128.STRONG.GPU") ? 16u : 0u;
          if (instr->getOperand(2)->type != InstrType::OperandType::PRED ||
              !candidate_width || instr->getSize() != candidate_width ||
              instr->getOperand(1)->nbytes != candidate_width)
            throw runtime_error("unvalidated LDG source-control form: " + op);
          const auto *control = instr->getOperand(2);
          source_predicate_type=1;source_predicate_number=control->u.pred.num;
          source_predicate_not=control->is_not ? 1 : 0;load_mask_width=candidate_width;
          if (source_predicate_number < 0 || source_predicate_number > 7 || control->is_neg || control->is_abs)
            throw runtime_error("unsupported LDG source predicate");
        }
        const string sass(instr->getSass());
        if (sass.find('\t') != string::npos || sass.find('\n') != string::npos)
          throw runtime_error("invalid LDG source instruction metadata text");
        load_mask_instructions_fp << function_id << '\t' << instr->getOffset() << '\t' << op << '\t'
          << instr->getSize() << "\t0\t" << source_predicate_type << '\t' << source_predicate_number << '\t' << source_predicate_not << '\t'
          << instr->getSize() << '\t' << sass << '\n';
        if (!load_mask_instructions_fp) throw runtime_error("cannot persist LDG source metadata");
        ++load_mask_static_records;
      }

      auto opcode_it = opcode_to_id_map.find(instr->getOpcode());
      if (opcode_it == opcode_to_id_map.end()) {
        int opcode_id = opcode_to_id_map.size();
        opcode_to_id_map[instr->getOpcode()] = opcode_id;
        id_to_opcode_map[opcode_id] = string(instr->getOpcode());
        opcode_it = opcode_to_id_map.find(instr->getOpcode());
      }

      sgsample::register_instruction(function_id,instr,opcode_it->second,mref_id,
        transfer_width,transfer_policy,load_mask_width);
      nvbit_insert_call(instr, "instrument_mem_fast", IPOINT_BEFORE);
      nvbit_add_call_arg_guard_pred_val(instr);
      nvbit_add_call_arg_const_val32(instr, (int)instr->getOffset());
      nvbit_add_call_arg_const_val32(instr, opcode_it->second);
      nvbit_add_call_arg_mref_addr64(instr, 0);
      nvbit_add_call_arg_const_val32(instr, mref_id);
      if (mref_id == 2)
        nvbit_add_call_arg_mref_addr64(instr, 1);
      else
        nvbit_add_call_arg_const_val64(instr, 0);
      nvbit_add_call_arg_const_val64(instr, (uint64_t)&channel_dev);
      nvbit_add_call_arg_const_val64(instr, (uint64_t)&pushed_record_count);
      nvbit_add_call_arg_const_val32(instr, space_hint);
      nvbit_add_call_arg_const_val32(instr, function_id);
      if (load_mask_width && source_predicate_number < 7)
        nvbit_add_call_arg_pred_reg(instr);
      else if (source_predicate_type == 1)
        nvbit_add_call_arg_pred_val_at(instr, source_predicate_number);
      else if (source_predicate_type == 2)
        nvbit_add_call_arg_upred_val_at(instr, source_predicate_number);
      else nvbit_add_call_arg_const_val32(instr, 1);
      nvbit_add_call_arg_const_val32(instr, source_predicate_not | ((load_mask_width && source_predicate_number < 7) ? ((source_predicate_number + 1) << 8) : 0));
      // Policy 2 is an internal LDG operand packet kind, never a cache policy.
      // Preserve the validated instrumentation ABI and packet layout.
      nvbit_add_call_arg_const_val32(instr, transfer_width ? transfer_width : load_mask_width);
      nvbit_add_call_arg_const_val32(instr, load_mask_width ? 2u : transfer_policy);
      nvbit_add_call_arg_const_val64(instr, (uint64_t)&sg_sample_cta_count);
      nvbit_add_call_arg_const_val64(instr, (uint64_t)sg_sample_cta_ids);
      cnt++;
    }
  }
}
