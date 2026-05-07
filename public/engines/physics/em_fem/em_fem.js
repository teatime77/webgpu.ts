init_potential();
while(true) {
    solve_potential();
    build_surface();
    render_surface();
    swapPingPong(Potential);
    yield;
}
