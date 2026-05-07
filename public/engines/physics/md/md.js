InitParticles();
RenderParticles();

swapPingPong(ParticlePos, ParticleVel);
yield;

while(true) {
    MdStep();
    RenderParticles();

    swapPingPong(ParticlePos, ParticleVel);
    yield;
}
