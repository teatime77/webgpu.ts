InitPsi();

NormPartialA();
NormTotalA();
NormApplyA();

RenderPsi();

swapPingPong(Psi);
yield;

while(true) {
    Rk2Mid();
    Rk2Finish();

    NormPartialB();
    NormTotalB();
    NormApplyB();

    RenderPsi();

    swapPingPong(Psi);
    yield;
}
