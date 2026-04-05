// todo: implement generic object detector struct, with bon builder
// the following params:
// * promptable/prompt_free (see structs.rs, prompt_free_detector.rs and promptable_detector.rs)
// * include_mask (det vs seg) (see structs.rs & model_manager.rs)
// * model scale (see structs.rs)
//
// only instantiate the required detector, promptable or prompt free, based on whether promptable/prompt_free was passed
// constructable only via from_hf builder method