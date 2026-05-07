init_temperature();
while(true) {
    fem_step();
    build_surface();
    render_surface();
    swapPingPong(Temperature);
    yield;
}
