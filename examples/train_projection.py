"""Train with a selectable loss; synthetic smoke demo or your own NPY inputs.

python examples/train_projection.py --loss smtl --geometry 'sphere*torus'
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from manifold_studio.cli import parse_geometry
from manifold_studio.losses import available_losses, make_loss, fixed_spectral_basis
from manifold_studio.training import TrainableProjector
from manifold_studio import neighbor_preservation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loss', choices=available_losses(), default='geodesic_triplet')
    parser.add_argument('--geometry', type=parse_geometry, default=parse_geometry('sphere'))
    parser.add_argument('--metric', choices=['intrinsic', 'ambient'], default=None)
    parser.add_argument('--input', help='Training embeddings N x D in .npy; limit 512 rows')
    parser.add_argument('--triplets', help='Integer training triplet indices T x 3 in .npy')
    parser.add_argument('--query', help='Query vector D or 1 x D in .npy for spectral losses')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--margin', type=float, default=0.3)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--gamma', type=float, default=0.3)
    parser.add_argument('--out', default='outputs/trained')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    if args.input:
        x = torch.tensor(np.load(args.input, allow_pickle=False), dtype=torch.float32)
        triplets = np.load(args.triplets, allow_pickle=False) if args.triplets else None
        query = torch.tensor(np.load(args.query, allow_pickle=False), dtype=x.dtype).reshape(1, -1) if args.query else None
    else:
        labels = torch.arange(3).repeat_interleave(16)
        centers = torch.randn(3, 12)
        x = centers[labels] + 0.35*torch.randn(48, 12)
        triplets = torch.tensor([[i, (i//16)*16+(i+1)%16, (i+16)%48] for i in range(48)])
        query = centers[:1]
    loss = make_loss(args.loss, args.geometry, metric=args.metric, margin=args.margin,
                     alpha=args.alpha, beta=args.beta, gamma=args.gamma)
    need_triplets = args.loss.endswith('triplet') or args.loss == 'smtl'
    spectral = args.loss in ('query_energy_separation', 'energy_concentration') or (args.loss == 'smtl' and (args.alpha > 0 or args.beta > 0))
    if need_triplets and triplets is None:
        parser.error('This loss requires --triplets with --input')
    if spectral and query is None:
        parser.error('This loss requires --query with --input')
    basis = fixed_spectral_basis(x) if spectral else None
    model = TrainableProjector(x.shape[1], args.geometry).fit_normalization(x)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    history = []
    for step in range(args.steps + 1):
        result = loss(model(x), triplets=triplets, source=x,
                      query_parameters=model(query) if spectral else None, basis=basis)
        history.append({'step': step, **result.detached()})
        if step == args.steps:
            break
        optimizer.zero_grad()
        result.total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
    model.eval().fit_display(x)
    embedding = model.transform(x)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    embedding.save(out/'embeddings.npz')
    embedding.export_csv(out/'coordinates.csv')
    model.save(out/'trained_projector.npz')
    report = {'loss': args.loss, 'metric': loss.metric, 'geometry': args.geometry.spec(),
              'weights': {'alpha': args.alpha, 'beta': args.beta, 'gamma': args.gamma},
              'margin': args.margin, 'seed': args.seed, 'steps': args.steps,
              'input_kind': 'user training embeddings' if args.input else 'synthetic smoke demo',
              'evaluation': 'Training objective only; no semantic accuracy or held-out generalization claim',
              'neighbors': neighbor_preservation(x.numpy(), embedding, metric=loss.metric),
              'history': history}
    (out/'training.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'output': str(out), 'initial': history[0], 'final': history[-1]}, indent=2))


if __name__ == '__main__':
    main()
