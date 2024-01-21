//maintenant on commence restructuration
//hiiii
//hahahahahhhhhahahah
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <cblas.h>

//
typedef float              f32;
typedef double             f64;
typedef unsigned long long u64;

//
typedef struct particle_s {

  f32* x;
  f32* y;
  f32* z;
  f32* vx;
  f32* vy;
  f32* vz;
  
} particle_t;

u64 i = 0;


//
void init(particle_t* p, u64 n)
{
const f32 invRAND_MAX = 1.0f / (f32)RAND_MAX;
  //#pragma omp parallel for
  for (i = 0; i < n; i++)
    {
      //
      u64 r1 = (u64)rand(), r2 = (u64)rand();
      f32 sign = (r1 > r2) ? 1 : -1;
      
      //
      p->x[i] = sign * (f32)rand() *invRAND_MAX;
      p->y[i] = (f32)rand() *invRAND_MAX;
      p->z[i] = sign * (f32)rand() *invRAND_MAX;

      //
      p->vx[i] = (f32)rand() *invRAND_MAX;
      p->vy[i] = sign * (f32)rand() *invRAND_MAX;
      p->vz[i] = (f32)rand() *invRAND_MAX;
    }
}

//
void move_particles(particle_t* p, const f32 dt, u64 n)
{

  //Used to avoid division by 0 when comparing a particle to itself
  const f32 softening = 1e-20;
  
  //For all particles
  #pragma omp parallel for
  for (i = 0; i < n; i++)
    {
      //
      f32 fx = 0.0, fy = 0.0, fz = 0.0;
      
      //
      const f32 px = p->x[i], py = p->y[i], pz = p->z[i];

	
      //Newton's law: 17 FLOPs (Floating-Point Operations) per iteration
      //#pragma omp parallel for schedule(static) reduction(+:fx, fy, fz)
      for (u64 j = 0; j < n; j++)
	{ 
	  //3 FLOPs (Floating-Point Operations) 
	  const f32 dx = p->x[j] - px, dy = p->y[j] - py, dz = p->z[j] - pz; //1 (sub)
	  //const f32 dy = p->y[j] - py; //2 (sub)
	  //const f32 dz = p->z[j] - pz; //3 (sub)

	  //Compute the distance between particle i and j: 6 FLOPs
	  const f32 d_2 = (dx * dx) + (dy * dy) + (dz * dz) + softening; //9 (mul, add)

	  //2 FLOPs (here, we consider pow to be 1 operation)
	  const f32 d_3_over_2 = d_2 * sqrt(d_2); //11 (pow, div)
	  
	  //Calculate net force: 6 FLOPs
	  fx += dx / d_3_over_2; //13 (add, div)
	  fy += dy / d_3_over_2; //15 (add, div)
	  fz += dz / d_3_over_2; //17 (add, div)
	}

      //Update particle velocities using the previously computed net force: 6 FLOPs 
      p->vx[i] += dt * fx; //19 (mul, add)
      p->vy[i] += dt * fy; //21 (mul, add)
      p->vz[i] += dt * fz; //23 (mul, add)
    }

  //Update positions: 6 FLOPs
  #pragma omp parallel for
  for (i = 0; i < n; i++)
    {
      p->x[i] += dt * p->vx[i];
      p->y[i] += dt * p->vy[i];
      p->z[i] += dt * p->vz[i];
    }
}



//
void write_positions(const char *f, particle_t* p, u64 n)
{
  FILE *file = fopen(f, "w");
  if (file != NULL)
  {
    for (i = 0; i < n; i++)
    {
      fprintf(file, "%e %e %e\n", p->x[i], p->y[i], p->z[i]);
    }
    fclose(file);
  }
  else
  {
    printf("Erreur lors de l'ouverture du fichier %s\n", f);
  }
}


void read_positions(const char *filename, particle_t* p_ref, u64 n)
{
  FILE *file = fopen(filename, "r");
  if (file != NULL)
  {
    for (i = 0; i < n; i++)
    {
      if (fscanf(file, "%f %f %f", &(p_ref->x[i]), &(p_ref->y[i]), &(p_ref->z[i])) != 3)
      {
        printf("Erreur lors de la lecture du fichier %s\n", filename);
        break;
      }
    }
    fclose(file);
  }
  else
  {
    printf("Erreur lors de l'ouverture du fichier %s\n", filename);
  }
}

//stabl num
      f64 compute_delta(particle_t* p_ref, particle_t* p, u64 n)
	{ 
	  f64 delta = 0.0;
	  #pragma omp parallel for
	  for (i = 0; i < n; i++){
	    delta += fabs(p_ref->x[i] - p->x[i]);
	    delta += fabs(p_ref->y[i] - p->y[i]);
	    delta += fabs(p_ref->z[i] - p->z[i]);
	}

	  delta /= (f64)(3 * n);

	  return delta;
	}




//
int main(int argc, char **argv)
{
  //Number of particles to simulate
  const u64 n = (argc > 1) ? atoll(argv[1]) : 16384;

  //Number of experiments
  const u64 steps= 13;

  //Time step
  const f32 dt = 0.01;

  //
  f64 rate = 0.0, drate = 0.0;

  //Steps to skip for warm up
  const u64 warmup = 3;
  
  //
  particle_t p;
  //
  p.x = (f32*)malloc(sizeof(f32) * n);
  p.y = (f32*)malloc(sizeof(f32) * n);
  p.z = (f32*)malloc(sizeof(f32) * n);
  p.vx = (f32*)malloc(sizeof(f32) * n);
  p.vy = (f32*)malloc(sizeof(f32) * n);
  p.vz = (f32*)malloc(sizeof(f32) * n);
  
  //
  particle_t p_ref;
  //
  p_ref.x = (f32*)malloc(sizeof(f32) * n);
  p_ref.y = (f32*)malloc(sizeof(f32) * n);
  p_ref.z = (f32*)malloc(sizeof(f32) * n);
  p_ref.vx = (f32*)malloc(sizeof(f32) * n);
  p_ref.vy = (f32*)malloc(sizeof(f32) * n);
  p_ref.vz = (f32*)malloc(sizeof(f32) * n);
  
    // Copy initial positions to the reference array
  read_positions("ref_positions.txt", &p_ref, n);
	
  //
  init(&p, n);

  const u64 s = sizeof(particle_t) * n;
  
  printf("\n\033[1mTotal memory size:\033[0m %llu B, %llu KiB, %llu MiB\n\n", s, s >> 10, s >> 20);
  
  //
  printf("\033[1m%5s %10s %10s %8s\033[0m\n", "Step", "Time, s", "Interact/s", "GFLOP/s"); fflush(stdout);
  
  //
  for (i = 0; i < steps; i++)
    {
      //Measure
      const f64 start = omp_get_wtime();

      move_particles(&p, dt, n);

      const f64 end = omp_get_wtime();

      //Number of interactions/iteration
      const f32 h1 = (f32)(n) * (f32)(n);

      //Number of GFLOPs
      //Innermost loop (Newton's law)   : 17 FLOPs x n (innermost trip count) x n (outermost trip count)
      //Velocity update (outermost body):  6 FLOPs x n (outermost trip count)
      //Positions update                :  6 FLOPs x n 
      const f32 h2 = (17.0 * h1 + 6.0 * (f32)n + 6.0 * (f32)n) * 1e-9;

      //Do not take warm up runs into account
      if (i >= warmup)
	{
	  rate += h2 / (f32)(end - start);
	  drate += (h2 * h2) / (f32)((end - start) * (end - start));
	}
      
      //
      printf("%5llu %10.3e %10.3e %8.1f %s\n",
	     i,
	     (end - start),
	     h1 / (end - start),
	     h2 / (end - start),
	     (i < warmup) ? "(warm up)" : "");
      
      fflush(stdout);
    }
  
  
      //
    char f[256];
    
    sprintf(f, "positions_at_end_simulation_neuf-Ofast-clang.txt");
    write_positions(f, &p, n);
    
    //
    f64 delta = compute_delta(&p_ref, &p, n);
    
  
  
  //Average GFLOPs/s
  rate /= (f64)(steps - warmup);

  //Deviation in GFLOPs/s
  drate = sqrt(drate / (f64)(steps - warmup) - (rate * rate));
  
  printf("-----------------------------------------------------\n");
  printf("\033[1m%s %4s \033[42m%10.1lf +- %.1lf GFLOP/s\033[0m\n",
	 "Average performance:", "", rate, drate);
  printf("-----------------------------------------------------\n");
    
          printf("la stabilité numérique delta est : %.7f\n",delta);
  //
  free(p.x);
  free(p.y);
  free(p.z);
  free(p.vx);
  free(p.vy);
  free(p.vz);

  //
  return 0;
}
