from __future__ import print_function, absolute_import


__all__ = ["accuracy"]


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    #print('output:', output)
    #print('target:', target)
    _, pred = output.topk(maxk, 1, True, True)

    pred_label = pred
    #print('1:', pred)

    pred = pred.t()
    #print('2:', pred)
    #print('1:', target)

    correct = pred.eq(target.view(1, -1).expand_as(pred))

    # print('correct:', correct)

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
        
    return res, pred_label, pred, target.unsqueeze(0)
