init_fields();

swapPingPong(Velocity, Dye);

yield;

while(true) {
    step_velocity();
    step_dye();
    build_surface();
    render_surface();
    swapPingPong(Velocity, Dye);
    yield;
}
