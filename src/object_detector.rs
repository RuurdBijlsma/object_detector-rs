// todo: implement generic object detector struct, with bon builder
// the following params:
// * promptable/prompt_free (see structs.rs)
// * include_mask (det vs seg) (see structs.rs & model_manager.rs)
// * model scale (see structs.rs)
//
// only instantiate the required detector, propmtable or prompt free, based on wether promptable/prompt_free was passed