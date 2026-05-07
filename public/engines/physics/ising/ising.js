InitSpins();
RenderSpins();

swapPingPong(SpinBuffer);
yield;

while(true) {
    UpdateSpins();
    RenderSpins();

    swapPingPong(SpinBuffer);
    yield;
}
