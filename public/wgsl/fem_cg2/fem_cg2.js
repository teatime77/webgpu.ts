init_positions();
init_elements();
init_b();

while(true) {
    cg_init();
    calc_rho();

    for(const i of range(metadata.cg_iters)) {
        clear_q();
        apply_A();
        calc_p_q();
        update_x_r();
        calc_new_rho();
        update_p();
    }

    render_mesh();
    yield;
}
