InitParticles();
RenderParticles();

swapPingPong(ParticlePos, ParticleVel);
yield;

while(true){
    UpdateParticles();
    RenderParticles();

    swapPingPong(ParticlePos, ParticleVel);
    yield;
}