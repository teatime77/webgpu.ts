InitParticles();
RenderParticles();

swapPingPong(ParticlePos, ParticleVel);
yield;

while(true){
    UpdateParticles();
    ClearGrid();
    BuildGrid();
    ResolveCollisions();
    RenderParticles();

    swapPingPong(ParticlePos, ParticleVel);
    yield;
}